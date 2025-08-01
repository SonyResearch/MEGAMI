
# adapted from  https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/dit.py

import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class OneHotPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your one-hot positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = F.one_hot(pos, num_classes=self.max_seq_len).to(x.dtype)
        return pos_emb


class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        pos_emb_strategy="concatenation",
        pos_emb_dim=None,
        pos_emb_type="one-hot",
        pos_emb_crossattn_strategy="concatenation",
        pos_emb_crossattn_dim=None,
        pos_emb_crossattn_type="one-hot",
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            #self.to_cond_embed = nn.Sequential(
            #    nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
            #    nn.SiLU(),
            #    nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            #)
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if pos_emb_strategy == "concatenation":

            assert pos_emb_dim is not None, "pos_emb_dim must be specified if pos_emb_strategy is concatenation"
            if pos_emb_type == "one-hot":
                # One-hot positional embeddings
                self.pos_emb = OneHotPositionalEmbedding(pos_emb_dim)
                self.extra_dim = pos_emb_dim

                def concat_pos_emb(x, pos=None, seq_start_pos=None):
                    pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos)
                    assert pos_emb.shape[-1] == pos_emb_dim, f"pos_emb shape mismatch: {pos_emb.shape[-1]} != {pos_emb_dim}"
                    assert pos_emb.shape[-2] == x.shape[-2], f"pos_emb sequence length mismatch: {pos_emb.shape[-2]} != {x.shape[-2]}"

                    if pos_emb.ndim == 2:
                        pos_emb = pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
                    else:
                        assert pos_emb.ndim == 3, f"pos_emb must be 2D or 3D, got {pos_emb.ndim}"
                        assert pos_emb.shape[0] == x.shape[0], f"pos_emb batch size mismatch: {pos_emb.shape[0]} != {x.shape[0]}"

                    return torch.cat((x, pos_emb), dim=-1)
                
                self.pos_emb_fn = concat_pos_emb
                self.remove_pos_emb = lambda x: x[..., :-pos_emb_dim] if pos_emb_dim > 0 else x

            else:
                raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}")
        else:
            raise ValueError(f"Unknown pos_emb_strategy: {pos_emb_strategy}")
        

        if pos_emb_crossattn_strategy == "concatenation":
            assert pos_emb_crossattn_dim is not None, "pos_emb_crossattn_dim must be specified if pos_emb_crossattn_strategy is concatenation"
            if pos_emb_crossattn_type == "one-hot":
                # One-hot positional embeddings for cross-attention
                self.pos_emb_crossattn = OneHotPositionalEmbedding(pos_emb_crossattn_dim)
                self.crossattn_extra_dim = pos_emb_crossattn_dim


                def concat_pos_emb_crossattn(x, pos=None, seq_start_pos=None):
                    pos_emb = self.pos_emb_crossattn(x, pos=pos, seq_start_pos=seq_start_pos)
                    assert pos_emb.shape[-1] == pos_emb_crossattn_dim, f"pos_emb shape mismatch: {pos_emb.shape[-1]} != {pos_emb_crossattn_dim}"
                    assert pos_emb.shape[-2] == x.shape[-2], f"pos_emb sequence length mismatch: {pos_emb.shape[-2]} != {x.shape[-2]}"

                    if pos_emb.ndim == 2:
                        pos_emb = pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
                    else:
                        assert pos_emb.ndim == 3, f"pos_emb must be 2D or 3D, got {pos_emb.ndim}"
                        assert pos_emb.shape[0] == x.shape[0], f"pos_emb batch size mismatch: {pos_emb.shape[0]} != {x.shape[0]}"



                    return torch.cat((x, pos_emb), dim=-1)

                self.pos_emb_crossattn_fn = concat_pos_emb_crossattn

            else:
                raise ValueError(f"Unknown pos_emb_type: {pos_emb_crossattn_type}")


        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            from networks.transformer_v3 import ContinuousTransformer

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                num_heads= num_heads,
                dim_in=dim_in + pos_emb_dim,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim+ pos_emb_crossattn_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
             
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)




    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        t=t.squeeze(1)


        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]


        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
       
        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        mask=None,
        return_info=False,
        **kwargs):


        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        #x= rearrange(x, "b  c t -> b  t c")
        #shape of contecxt is already [B, N, T, C] so no need to rearrange 

        #orig_shape = x.shape

        #x = rearrange(x, "b t c -> b c t")
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        x= self.pos_emb_fn(x)

        #if cross_attn_cond is not None:
        #    cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        cross_attn_cond= self.pos_emb_crossattn_fn(cross_attn_cond) 

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond, 
            cross_attn_cond_mask=cross_attn_cond_mask, 
            input_concat_cond=input_concat_cond, 
            global_embed=global_embed, 
            mask=mask,
            return_info=return_info,
            **kwargs
        )