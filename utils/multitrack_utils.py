
import einops
import time
import torch


def multitrack_batched_processing( x, taxonomy=None,function=None, class_dependent=False, masks=None, number_outputs=1):
    """
    x: tensor of shape [B, N, C, T] where B is the batch size, N is the number of tracks, C is the number of channels and T is the length of the audio
    taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is
    function: function to apply to each track, it should take a tensor of shape [B, C, T] and a list of taxonomies as input and return a tensor of the same shape

    This function reshapes the input tensor x to a 2D tensor of shape [B*N, C, T] and applies the function to each track independently. It then reshapes the output back to the original shape [B, N, C, T].
    """

    assert not class_dependent, "this function needs an effect randomizer that is not class dependent (although it needs the taxonomy list), use simulate_effects instead"

    if masks is None:
        masks = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)


    original_shape=x.shape

    x_masked, taxonomy_reshaped = forward_reshaping(x, taxonomy, masks)

    func_out=function(x_masked, taxonomy_reshaped)

    if number_outputs==1:

        output_shape = (original_shape[0], original_shape[1], func_out.shape[-2],func_out.shape[-1])
    
        out = torch.zeros(output_shape, dtype=func_out.dtype, device=func_out.device)
    
        # Create a counter to keep track of where we are in x_emb
        emb_idx = 0
    
        for b in range(original_shape[0]):
            for n in range(original_shape[1]):
                if masks[b, n]:
                    out[b, n] = func_out[emb_idx]
                    emb_idx += 1
    
        return out
    elif number_outputs>1:

        outs=()

        for i in range(number_outputs):
            func_out_i = func_out[i]
            output_shape= (original_shape[0], original_shape[1], func_out_i.shape[-2],func_out_i.shape[-1])

            out = torch.zeros(output_shape, dtype=func_out_i.dtype, device=func_out_i.device)

            # Create a counter to keep track of where we are in x_emb
            emb_idx = 0
            for b in range(original_shape[0]):
                for n in range(original_shape[1]):
                    if masks[b, n]:
                        out[b, n] = func_out_i[emb_idx]
                        emb_idx += 1

            outs += (out,)

        return outs



def forward_reshaping( x, taxonomy, masks=None):
    """
    x: tensor of shape [B, N, C, T] where B is the batch size, N is the number of tracks, C is the number of channels and T is the length of the audio
    taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is
    function: function to apply to each track, it should take a tensor of shape [B, C, T] and a list of taxonomies as input and return a tensor of the same shape

    This function reshapes the input tensor x to a 2D tensor of shape [B*N, C, T] and applies the function to each track independently. It then reshapes the output back to the original shape [B, N, C, T].
    """

    if masks is None:
        masks = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)

    original_shape=x.shape
    x_reshaped=einops.rearrange(x, "b n c t -> (b n) c t")  #flatten the batch and number of tracks

    mask_reshaped=einops.rearrange(masks, "b n -> (b n)") if masks is not None else None

    x_masked=x_reshaped[mask_reshaped]

    taxonomy_reshaped=[]
    if taxonomy is not None:
        for b in range(original_shape[0]):
            for n in range(original_shape[1]):
                if masks[b, n]:
                    taxonomy_reshaped.append(taxonomy[b][n])

    return x_masked, taxonomy_reshaped
