
import einops
import time
import torch

def simulate_effects_old( x, cluster, taxonomy, effect_randomizer_C0, effect_randomizer_C1, masks=None):
    """
    x: tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
    cluster: tensor of shape [B] where B is the batch size
    taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is the tracks, each taxonomy is a string 
    effect_randomizer_C0: dictionary of EffectRandomizer objects for cluster 0
    effect_randomizer_C1: dictionary of EffectRandomizer objects for cluster 1
    """



    #first separate x in clusters
    C0= (cluster==0)
    C1= (cluster==1)

    y=x.clone() #initialize y with x

    indices_C0= torch.where(C0)[0]  # indices of the samples in cluster C0
    indices_C1= torch.where(C1)[0]  # indices of the samples in cluster C1

    a=time.time()
    if (C0).any():
        for k in effect_randomizer_C0.keys():
            indices_k=torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)  # initialize a tensor of shape [B, N] with False
            for index in indices_C0:
                for i, t in enumerate(taxonomy[index]):
                    if t == k:
                        indices_k[index, i] = True
            y[indices_k] = effect_randomizer_C0[k].forward(x[indices_k])  # apply the effect randomizer to the selected tracks

    if (C1).any():
        for k in effect_randomizer_C1.keys():
            indices_k=torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)  # initialize a tensor of shape [B, N] with False
            for index in indices_C1:
                for i, t in enumerate(taxonomy[index]):
                    if t == k:
                        indices_k[index, i] = True
            y[indices_k] = effect_randomizer_C1[k].forward(x[indices_k])  # apply the effect randomizer to the selected tracks

    return y

def simulate_effects( x, cluster, taxonomy, effect_randomizer_C0, effect_randomizer_C1, masks=None):
    """
    x: tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
    cluster: tensor of shape [B] where B is the batch size
    taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is the tracks, each taxonomy is a string 
    effect_randomizer_C0: dictionary of EffectRandomizer objects for cluster 0
    effect_randomizer_C1: dictionary of EffectRandomizer objects for cluster 1
    """

    def apply_effect_randomizer(x, taxonomy=None, cluster=0):
        """
        Apply the effect randomizer to the selected tracks.
        """
        y= x.clone()  # initialize y with x
        if cluster==0:
            for k in effect_randomizer_C0.keys():
                indices_k=torch.zeros((x.shape[0]), dtype=torch.bool, device=x.device)  # initialize a tensor of shape [B, N] with False
                for i, t in enumerate(taxonomy):
                    if t == k:
                        indices_k[i] = True
                if indices_k.any():
                    y[indices_k] = effect_randomizer_C0[k].forward(x[indices_k])
            return y
        elif cluster==1:
            for k in effect_randomizer_C1.keys():
                indices_k=torch.zeros((x.shape[0]), dtype=torch.bool, device=x.device)  # initialize a tensor of shape [B, N] with False
                for i, t in enumerate(taxonomy):
                    if t == k:
                        indices_k[i] = True
                if indices_k.any():
                    y[indices_k] = effect_randomizer_C1[k].forward(x[indices_k])
            return y
        else:
            raise ValueError("cluster must be 0 or 1")



    #first separate x in clusters
    C0= (cluster==0)
    C1= (cluster==1)

    y=x.clone() #initialize y with x

    indices_C0= torch.where(C0)[0]  # indices of the samples in cluster C0
    indices_C1= torch.where(C1)[0]  # indices of the samples in cluster C1

    a=time.time()
    if (C0).any():
        x_C0 = x[C0]  # select the samples in cluster C0
        taxonomies_C0 = [taxonomy[i] for i in indices_C0]  # select the taxonomies for the samples in cluster C0
        mask_C0 = masks[C0] if masks is not None else None  # select the masks for the samples in cluster C0
        #print("C0",taxonomies_C0)
        y_C0=multitrack_batched_processing(x_C0, taxonomy=taxonomies_C0, function=lambda x, t: apply_effect_randomizer(x,t,0), class_dependent=False, masks=mask_C0)
        y[C0] = y_C0  # apply the effect randomizer to the selected tracks

    if (C1).any():
        x_C1= x[C1]  # select the samples in cluster C1
        taxonomies_C1= [taxonomy[i] for i in indices_C1]  # select the taxonomies for the samples in cluster C1
        masks_C1 = masks[C1] if masks is not None else None  # select the masks for the samples in cluster C1
        #print("C1",taxonomies_C1)
        y_C1=multitrack_batched_processing(x_C1, taxonomy=taxonomies_C1, function=lambda x, t: apply_effect_randomizer(x,t, 1), class_dependent=False, masks=masks_C1) 
        y[C1] = y_C1  # apply the effect randomizer to the selected

    return y

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

