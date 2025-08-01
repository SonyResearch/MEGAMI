
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mapper_architectures import TCNModel

from fx_model.processors.compexp import prepare_compexp_parameters, compexp_functional
from fx_model.processors.fdn import prepare_FDN_parameters, fdn_functional
from fx_model.processors.peq import prepare_PEQ_parameters, peq_functional, prepare_PEQ_FDN_parameters, peq_FDN_functional
from fx_model.processors.transformations import  SmoothingCoef, MinMax 

import math

def panning( x, pan_param):
        angle = pan_param.view(-1,1) * torch.pi * 0.5
        amp = torch.concat([angle.cos(), angle.sin()],dim=1).view(-1,2, 1) * math.sqrt(2)  # Normalize to avoid gain increase
        return x * amp

class DiffVoxFx_pipeline(nn.Module):

    def __init__(self, 
        sample_rate=44100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ):
        super(DiffVoxFx_pipeline, self).__init__()

        self.sample_rate = sample_rate
        self.total_num_params=0

        ## PEQ parameters
        params_PEQ_optimizable, params_PEQ_non_optimizable, dict_transformations = prepare_PEQ_parameters(sample_rate, device=device)

        for key, param in params_PEQ_optimizable.items():
            self.total_num_params += param.numel()

        self.params_PEQ_non_optimizable = params_PEQ_non_optimizable
        self.PEQ_transformations = dict_transformations

        ## Compressor and Expander parameters
        params_CompExp_optimizable, params_CompExp_non_optimizable, dict_transformations = prepare_compexp_parameters(sample_rate, device=device)
        for key, param in params_CompExp_optimizable.items():
            self.total_num_params += param.numel()


        self.params_CompExp_non_optimizable = params_CompExp_non_optimizable
        self.CompExp_transformations = dict_transformations



        ## PEQ parameters for FDN
        params_PEQ_FDN_optimizable, params_PEQ_FDN_non_optimizable, dict_transformations = prepare_PEQ_FDN_parameters(sample_rate, device=device)

        for key, param in params_PEQ_FDN_optimizable.items():
            self.total_num_params += param.numel()

        self.params_PEQ_FDN_non_optimizable = params_PEQ_FDN_non_optimizable
        self.PEQ_FDN_transformations = dict_transformations


        ## FDN parameters
        params_FDN_optimizable, params_FDN_non_optimizable, dict_transformations = prepare_FDN_parameters(sample_rate, device=device,  ir_duration = 6.0)
        for key, param in params_FDN_optimizable.items():
            #print("FDN param reference", key, param, param.shape, param.numel())
            self.total_num_params += param.numel()

        self.params_FDN_non_optimizable = params_FDN_non_optimizable
        self.FDN_transformations = dict_transformations

        ## panning parameter
        self.pan_param_transformation = SmoothingCoef()
        self.total_num_params += 1  # For the panning parameter

        ## gain parameter
        self.gain_param_transformation = MinMax(min=-20, max=20)  # Transformation for gain parameter
        self.total_num_params += 1  # For the gain parameter

        def param_processor(param_tensor):
            """
            param_tensor: shape (batch_size, num_params)
            """
            #print(param_tensor.shape, "param_tensor shape", self.total_num_params, "total_num_params")
            params_dict = {}
            params_dict['PEQ'] ={}

            index = 0

            for key, param_ref in params_PEQ_optimizable.items():
                #print("param ref",key, param_ref.shape)
                #print("param", key, param_ref.shape, param_ref.numel(), "index", index)
                param= param_tensor[:, index:index + param_ref.numel()]
                #print(f"Processing PEQ parameter {key} with shape {param.shape}")
                #if param_ref.shape is empty, then do view(-1, 1)
                if param_ref.shape == ():
                    params_dict['PEQ'][key] = param.view(-1, 1)
                else:
                    params_dict['PEQ'][key] = param.view(-1, *param_ref.shape)

                index += param_ref.numel()

            params_dict['CompExp'] ={}
            for key, param_ref in params_CompExp_optimizable.items():
                #print("param ref",key, param_ref.shape)
                #print("param", key, param_ref.shape, param_ref.numel(), "index", index  )
                param= param_tensor[:, index:index + param_ref.numel()]
                #print(f"Processing CompExp parameter {key} with shape {param.shape}")
                if param_ref.shape == ():
                    params_dict['CompExp'][key] = param.view(-1, 1)
                else:
                    params_dict['CompExp'][key] = param.view(-1, *param_ref.shape)

                index += param_ref.numel()
            
            params_dict['FDN'] ={}
            for key, param_ref in params_FDN_optimizable.items():
                #print("param ref",key, param_ref.shape)
                #print("param reference", param_ref," numel", param_ref.numel(), "index", index, "total num params", self.total_num_params, param_ref.shape)
                param= param_tensor[:, index:index + param_ref.numel()]
                #print(f"Processing FDN parameter {key} with shape {param.shape}")
                if param_ref.shape == ():
                    params_dict['FDN'][key] = param.view(-1, 1)
                else:
                    params_dict['FDN'][key] = param.view(-1, *param_ref.shape)

                index += param_ref.numel()
            
            params_dict['PEQ_FDN'] ={}
            for key, param_ref in params_PEQ_FDN_optimizable.items():
                #print("param ref",key, param_ref.shape)
                #print("param", key, param_ref.shape, param_ref.numel(), "index", index)
                param= param_tensor[:, index:index + param_ref.numel()]
                #print(f"Processing PEQ_FDN parameter {key} with shape {param.shape}")
                if param_ref.shape == ():
                    params_dict['PEQ_FDN'][key] = param.view(-1, 1)
                else:
                    params_dict['PEQ_FDN'][key] = param.view(-1, *param_ref.shape[1:])

                index += param_ref.numel()
            
            params_dict['pan'] = param_tensor[:, index:index + 1].view(-1, 1)
            index += 1

            params_dict['gain_dB'] = param_tensor[:, index:index + 1].view(-1, 1)

            index+=1 

            assert index == self.total_num_params, f"Expected {self.total_num_params} parameters, but got {index}"

            return params_dict
        
        self.param_processor = param_processor


    

    def forward(self, x, est_param):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        est_param: shape (batch_size, num_params)
        """

        #start= time.time()

        est_params_dict= self.param_processor(est_param)

        PEQ_params= est_params_dict['PEQ']
        CompExp_params= est_params_dict['CompExp']
        FDN_params= est_params_dict['FDN']
        PEQ_FDN_params= est_params_dict['PEQ_FDN']
        pan_param= est_params_dict['pan']
        gain_dB_param= est_params_dict['gain_dB']


        transformed_params_PEQ={}
        for key, param in PEQ_params.items():
            #print("param ",key, param.shape)
            param_transformed= self.PEQ_transformations[key](param)
            #print("param transformed ",key, param_transformed.shape)
            transformed_params_PEQ[key] = param_transformed
        
        x= peq_functional(x, **transformed_params_PEQ, **self.params_PEQ_non_optimizable)

        transformed_params_CompExp={}
        for key, param in CompExp_params.items():
            #print("param ",key, param.shape)
            param_transformed= self.CompExp_transformations[key](param)
            transformed_params_CompExp[key] = param_transformed
        
        x= compexp_functional(x, **transformed_params_CompExp, **self.params_CompExp_non_optimizable)

        transformed_params_FDN={}
        for key, param in FDN_params.items():
            #print("param ",key, param.shape)
            param_transformed= self.FDN_transformations[key](param)
            #print("param transformed ",key, param_transformed.shape)
            transformed_params_FDN[key] = param_transformed
        
        transformed_params_PEQ_FDN={}
        for key, param in PEQ_FDN_params.items():
            #print("param ",key, param.shape)
            param_transformed= self.PEQ_FDN_transformations[key](param)
            transformed_params_PEQ_FDN[key] = param_transformed
        
        def eq_fn(h):
            B, C1, C2, T = h.shape
            h= h.view(B, C1 * C2, T)
            h=peq_FDN_functional(h, **transformed_params_PEQ_FDN, **self.params_PEQ_FDN_non_optimizable)
            return h.view(B, C1, C2, T)

        x_to_fdn=x.repeat(1,2,1)
        x_fdn=fdn_functional(x_to_fdn, **transformed_params_FDN, **self.params_FDN_non_optimizable, eq=eq_fn)


        pan_param_transformed= self.pan_param_transformation(pan_param)
        x_pan=panning( x, pan_param_transformed)

        x = x_fdn + x_pan  # Add the panned signal to the original signal


        gain_dB= self.gain_param_transformation(gain_dB_param).view(-1,1,1)  # Ensure gain is broadcastable

        x= x * (10 ** (gain_dB / 20))  # Apply gain in dB

        return x

class DiffVoxParamEstimatorTCN(nn.Module):
    def __init__(self, 
                ninputs=1,
                cond_dim=192,
                sample_rate=44100,
                use_CLAP=False,
                CLAP_args=None,
                ):
        """
        Parameter estimator hardcoded for DiffVox effects pipeline.
        """

        super(DiffVoxParamEstimatorTCN, self).__init__()


        self.sample_rate = sample_rate

        self.fx_processors = {}

        self.fx_pipeline = DiffVoxFx_pipeline(sample_rate=sample_rate)

        total_num_param =  self.fx_pipeline.total_num_params
            
        ''' model architecture '''
        self.network = TCNModel(ninputs=ninputs, \
                                    noutputs=total_num_param, \
                                    nblocks=14, \
                                    dilation_growth=2, \
                                    kernel_size=15,\
                                    stride=1, \
                                    channel_width=128, \
                                    stack_size=15, \
                                    cond_dim=cond_dim, \
                                    causal=False,
                                    use_CLAP=use_CLAP, \
                                    CLAP_args=CLAP_args,
                                    )

    # network forward operation
    def forward(self, x, embedding):
        # embedding mapper
        est_param = self.network(x, embedding)
        est_param = est_param.mean(axis=-1)

        x=self.fx_pipeline(x, est_param)


        return x

from torchaudio.transforms import MelSpectrogram, MFCC
class LogMelSpectrogram(MelSpectrogram):
    def forward(self, waveform):
        return super().forward(waveform).add(1e-8).log()


class LogRMS(nn.Module):
    def forward(self, frame):
        return torch.log(frame.square().mean(-2, keepdim=True).sqrt() + 1e-8)


class LogCrest(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = LogRMS()

    def forward(self, frame):
        log_rms = self.rms(frame)
        return frame.abs().amax(-2, keepdim=True).add(1e-8).log() - log_rms


class LogSpread(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = LogRMS()

    def forward(self, frame):
        log_rms = self.rms(frame)
        return (frame.abs().add(1e-8).log() - log_rms).mean(-2, keepdim=True)

class DynamicParams(nn.Module):
    def __init__(self, frame_length, hop_length, center=False):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

        self.LogSpread = LogSpread()
        self.LogCrest = LogCrest()
        self.LogRMS = LogRMS()

    def forward(self, waveform):
        if self.center:
            waveform = F.pad(waveform, (self.frame_length // 2, self.frame_length // 2))
        frames= waveform.unfold(-1, self.frame_length, self.hop_length).transpose(-1, -2)

        log_spread = self.LogSpread(frames)
        log_crest = self.LogCrest(frames)
        log_rms = self.LogRMS(frames)

        return torch.cat([log_spread, log_crest, log_rms], dim=-2)

from typing import List
class MapAndMerge(nn.Module):
    def __init__(self, funcs: List[nn.Module], dim=-1):
        super().__init__()
        self.funcs = nn.ModuleList(funcs)
        self.dim = dim

    def forward(self, frame):
        return torch.cat([f(frame) for f in self.funcs], dim=self.dim)


class Frame(nn.Module):
    def __init__(self, frame_length, hop_length, center=False):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

    def forward(self, waveform):
        if self.center:
            waveform = F.pad(waveform, (self.frame_length // 2, self.frame_length // 2))
        return waveform.unfold(-1, self.frame_length, self.hop_length).transpose(-1, -2)

class DiffVoxParamEstimatorLogMelSpec(nn.Module):
    def __init__(self, 
                cond_dim=192,
                sample_rate=44100,
                ):
        """
        Parameter estimator hardcoded for DiffVox effects pipeline.
        """

        super(DiffVoxParamEstimatorLogMelSpec, self).__init__()

        self.fx_pipeline = DiffVoxFx_pipeline(sample_rate=sample_rate)


        self.feature_extractor = MapAndMerge([
                nn.Sequential(
                    Frame(frame_length=1024, hop_length=256, center=True),
                    MapAndMerge([LogRMS(), LogCrest(), LogSpread()], dim=-2)
                ),
                LogMelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80
                )
            ], dim=-2)
        

        # Flatten features
        self.flatten = nn.Flatten(start_dim=1, end_dim=-2)

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(83+cond_dim, 512, kernel_size=5, stride=1),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Conv1d(512, 512, kernel_size=5),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Conv1d(512, 768, kernel_size=5),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            
            nn.Conv1d(768, 1024, kernel_size=5),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1),
        )
        
        # Final layers
        self.final_flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(1024, self.fx_pipeline.total_num_params)

    
        self.sample_rate = sample_rate

        self.fx_processors = {}

        self.total_num_param =  self.fx_pipeline.total_num_params
            
        ''' model architecture '''
    
    def forward(self, x, embedding):
        # x shape: [batch, channels, time]
        
        # Extract features
        features = self.feature_extractor(x)


        
        # Flatten features
        flattened = self.flatten(features)

        embedding_expanded = embedding.unsqueeze(-1)  # Expand embedding to match feature dimensions
        embedding = embedding_expanded.expand(-1, -1, flattened.size(-1))

        flattened = torch.cat((flattened, embedding), dim=1)  # Concatenate embedding


        
        # Apply CNN layers
        conv_out = self.conv_layers(flattened)
        
        # Final flattening and linear layer
        flat = self.final_flatten(conv_out)
        est_param = self.fc(flat)
        
        y=self.fx_pipeline(x, est_param)
        return y






class DiffVoxParamEstimatorCLAPMLP(nn.Module):
    def __init__(self, 
                cond_dim=192,
                hidden_dim=512,
                dropout_rate=0.2,
                num_layers=3,
                sample_rate=44100,
                CLAP_args=None
                ):
        """
        Parameter estimator hardcoded for DiffVox effects pipeline.
        """

        super(DiffVoxParamEstimatorCLAPMLP, self).__init__()

        assert CLAP_args is not None, "CLAP_args must be provided for CLAP AE"
        from evaluation.feature_extractors import load_CLAP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CLAP_encoder= load_CLAP(CLAP_args, device=device)
        cond_dim+= 512
        def merge_CLAP_embeddings(x, emb):

                clap_embedding = CLAP_encoder(x, type="dry")
                #l2 normalize the clap embedding
                clap_embedding = F.normalize(clap_embedding, p=2, dim=-1)

                return torch.cat((emb, clap_embedding), dim=-1)
            
        self.merge_CLAP_embeddings = merge_CLAP_embeddings


        self.sample_rate = sample_rate

        self.fx_processors = {}

        self.fx_pipeline = DiffVoxFx_pipeline(sample_rate=sample_rate)

        self.total_num_param =  self.fx_pipeline.total_num_params
            
        ''' model architecture '''
        # Input layer
        self.input_fc = nn.Linear(cond_dim, hidden_dim)
        
        # Residual MLP layers
        self.fc_layers1 = nn.ModuleList()
        self.fc_layers2 = nn.ModuleList()
        self.bn_layers1 = nn.ModuleList()
        self.bn_layers2 = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.fc_layers1.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers1.append(nn.BatchNorm1d(hidden_dim))
            self.fc_layers2.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers2.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, self.total_num_param)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # network forward operation
    def forward(self, x, embedding):

        with torch.no_grad():
            embedding = self.merge_CLAP_embeddings(x, embedding)

        # Input layer
        h = self.input_fc(embedding)
        h = F.relu(h)
        
        # Residual blocks
        for i in range(len(self.fc_layers1)):
            identity = h
            
            # First linear + BN + ReLU
            h = self.fc_layers1[i](h)
            h = self.bn_layers1[i](h)
            h = F.relu(h)
            
            # Dropout
            h = self.dropout_layers[i](h)
            
            # Second linear + BN
            h = self.fc_layers2[i](h)
            h = self.bn_layers2[i](h)
            
            # Residual connection
            h = h + identity
            h = F.relu(h)
        
        # Output layer
        est_param = self.output_fc(h)
        
        x=self.fx_pipeline(x, est_param)


        return x



