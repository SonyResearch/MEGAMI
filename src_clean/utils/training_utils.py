
import math
import torch
import torchaudio
import numpy as np
import scipy.signal

def resample_batch(audio, fs, fs_target, length_target=None):

        device=audio.device
        dtype=audio.dtype
        B=audio.shape[0]
        #if possible resampe in a batched way
        #check if all the fs are the same and equal to 44100
        #print(fs_target)
        if fs_target==22050:
            if (fs==44100).all():
                 audio=torchaudio.functional.resample(audio, 2,1)
                 return audio[:, 0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 160*2,147)
                 return audio[:, 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                proc_batch=torch.zeros((B,length_target), device=device)
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    if f_s==44100:
                        #resample by 2
                        a=torchaudio.functional.resample(a, 2,1)
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 160*2,147)
                    elif f_s==22050:
                        pass
                    else:
                        print("WARNING, strange fs", f_s)
           
                    proc_batch[i]=a[0:length_target]
                return proc_batch
        elif fs_target==44100:
            if (fs==44100).all():
                 return audio[:, 0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 160,147)
                 return audio[:, 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                #B,C,L=audio.shape
                #proc_batch=torch.zeros((B,C,L), device=device)
                proc_batch=torch.zeros((B,length_target), device=device)
                #print("debigging resample batch")
                #print(audio.shape,fs.shape)
                #for i, (a, f_s) in enumerate(zip(audio, fs.tolist())): #I hope this shit wll not slow down everythingh
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    #print(i,a.shape,f_s)
                    if f_s==44100:
                        #resample by 2
                        pass
                    elif f_s==22050:
                        a=torchaudio.functional.resample(a, 1,2)
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 160,147)
                    elif f_s==96000:
                        a=torchaudio.functional.resample(a, 320, 147)
                    else:
                        print("WARNING, strange fs", f_s)
           

                    proc_batch[i]=a[...,0:length_target] 
                return proc_batch
        else:
            if (fs==44100).all():
                 audio=torchaudio.functional.resample(audio, 44100, fs_target)
                 return audio[...,0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 print("resampling 48000 to 16000", length_target, audio.shape)
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 48000,fs_target)
                 print(audio.shape)
                 return audio[..., 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                proc_batch=torch.zeros((B,length_target), device=device)
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    if f_s==44100:
                        #resample by 2
                        a=torchaudio.functional.resample(a, 44100,fs_target)
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 48000,fs_target)
                    else:
                        print("WARNING, strange fs", f_s)
           
                    proc_batch[i]=a[...,0:length_target] 
                return proc_batch

def load_state_dict( state_dict, network=None, ema=None, optimizer=None, log=True):
        '''
        utility for loading state dicts for different models. This function sequentially tries different strategies
        args:
            state_dict: the state dict to load
        returns:
            True if the state dict was loaded, False otherwise
        Assuming the operations are don in_place, this function will not create a copy of the network and optimizer (I hope)
        '''
        #print(state_dict)
        if log: print("Loading state dict")
        if log:
            print(state_dict.keys())
        #if there
        try:
            if log: print("Attempt 1: trying with strict=True")
            if network is not None:
                network.load_state_dict(state_dict['network'])
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])
            if ema is not None:
                ema.load_state_dict(state_dict['ema'])
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
        try:
            print("assuming the network was saved from a DDP model, and I forgot to add .module to the keys")
            #verify that the keys are the same but with .module removed
            if network is not None:
                network_state_dict = network.state_dict()
                for key in list(state_dict['network'].keys()):
                    print("checking", key)
                    if key.startswith('module.'):
                        new_key = key.replace('module.', '')
                        state_dict['network'][new_key] = state_dict['network'].pop(key)

                network.load_state_dict(state_dict['network'])
            
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])
            if ema is not None:
                ema.load_state_dict(state_dict['ema'])

            print("loaded state dict with .module removed from the keys")

            return True

        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)

        try:
            if log: print("Attempt 2: trying with strict=False")
            if network is not None:
                network.load_state_dict(state_dict['network'], strict=False)
            #we cannot load the optimizer in this setting
            #self.optimizer.load_state_dict(state_dict['optimizer'], strict=False)
            if ema is not None:
                ema.load_state_dict(state_dict['ema'], strict=False)
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
        try:
            if log: print("Attempt 3: trying with strict=False,but making sure that the shapes are fine")
            if ema is not None:
                ema_state_dict = ema.state_dict()
            if network is not None:
                network_state_dict = network.state_dict()
            i=0 
            if network is not None:
                for name, param in state_dict['network'].items():
                    if log: print("checking",name) 
                    if name in network_state_dict.keys():
                        if network_state_dict[name].shape==param.shape:
                                network_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
            network.load_state_dict(network_state_dict)
            if ema is not None:
                for name, param in state_dict['ema'].items():
                        if log: print("checking",name) 
                        if name in ema_state_dict.keys():
                            if ema_state_dict[name].shape==param.shape:
                                ema_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
     
            ema.load_state_dict(ema_state_dict)
     
            if i==0:
                if log: print("WARNING, no parameters were loaded")
                raise Exception("No parameters were loaded")
            elif i>0:
                if log: print("loaded", i, "parameters")
                return True

        except Exception as e:
            print(e)
            print("the second strict=False failed")


        try:
            if log: print("Attempt 4: Assuming the naming is different, with the network and ema called 'state_dict'")
            if network is not None:
                network.load_state_dict(state_dict['state_dict'])
            if ema is not None:
                ema.load_state_dict(state_dict['state_dict'])
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
                print("It failed 3 times!! but not giving up")
            #print the names of the parameters in self.network

        try:
            if log: print("Attempt 5: trying to load with different names, now model='model' and ema='ema_weights'")
            if ema is not None:
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                    ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)

        try:
            if log: print("Attempt 6: If there is something wrong with the name of the ema parameters, we can try to load them using the names of the parameters in the model")
            if ema is not None:
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)


        try:
            #assign the parameters in state_dict to self.network using a for loop
            print("Attempt 7: Trying to load the parameters one by one. This is for the dance diffusion model, looking for parameters starting with 'diffusion.' or 'diffusion_ema.'")
            if ema is not None:
                ema_state_dict = ema.state_dict()
            if network is not None:
                network_state_dict = ema.state_dict()
            i=0 
            if network is not None:
                for name, param in state_dict['state_dict'].items():
                    print("checking",name) 
                    if name.startswith("diffusion."):
                        i+=1
                        name=name.replace("diffusion.","")
                        if network_state_dict[name].shape==param.shape:
                            #print(param.shape, network.state_dict()[name].shape)
                            network_state_dict[name]=param
                            #print("assigning",name)
           
                network.load_state_dict(network_state_dict, strict=False)
           
            if ema is not None:
                for name, param in state_dict['state_dict'].items():
                    if name.startswith("diffusion_ema."): 
                        i+=1
                        name=name.replace("diffusion_ema.","")
                        if ema_state_dict[name].shape==param.shape:
                            if log:
                                    print(param.shape, ema.state_dict()[name].shape)
                            ema_state_dict[name]=param
           
                ema.load_state_dict(ema_state_dict, strict=False)
           
            if i==0:
                print("WARNING, no parameters were loaded")
                raise Exception("No parameters were loaded")
            elif i>0:
                print("loaded", i, "parameters")
                return True
        except Exception as e:
            if log:
                print(e)
        #try:
        # this is for the dmae1d mddel, assuming there is only one network
        if network is not None:
            network.load_state_dict(state_dict, strict=True)
        if ema is not None:
            ema.load_state_dict(state_dict, strict=True)
        return True

        #except Exception as e:
        #    if log:
        #        print(e)

        return False



def Gauss_smooth(X,f,Noct=1, smooth_filter=None):
    """
    based on https://github.com/IoSR-Surrey/MatlabToolbox/blob/4bff1bb2da7c95de0ce2713e7c710a0afa70c705/%2Biosr/%2Bdsp/smoothSpectrum.m
    Smooths the magnitude spectrum X using a Gaussian filter.

    Args:
        X (torch.Tensor): Input spectrum to be smoothed, shape (B, N,).
        f (torch.Tensor): Frequency bins corresponding to the spectrum, shape (N,).
        Noct (int, optional): Number of octaves for smoothing. Default is 1
    Returns:
        torch.Tensor: Smoothed spectrum, same shape as X.
    """

    def gauss_f(f_x,F,Noct):
        sigma = (F/Noct)/np.pi
        g = torch.exp(-(((f_x-F)**2)/(2*(sigma**2))))
        g = g/torch.sum(g)
        return g

    shape=X.shape

    x_oct = X.clone().view(-1, shape[-1])  # Initialize smoothed output
    if Noct>0:
        for i in range(1,len(f)):
            g = gauss_f(f,f[i],Noct)
            g=g.to(X.device).view(1, -1)
            x_oct[...,i] = torch.sum(g*X)


        x_oct = x_oct.clamp(min=0)  # Ensure non-negative values

    return x_oct.view(shape)  # Reshape back to original dimensions

def prepare_smooth_filter(f, Noct=1):
     f_matrix = f.unsqueeze(0)  # Shape: [1, N]
     F_matrix = f.unsqueeze(1)  # Shape: [N, 1]
     
     # Calculate sigma for each center frequency
     sigma = (F_matrix / Noct) / np.pi  # Shape: [N, 1]
     
     # Calculate Gaussian weights for all frequency pairs at once
     g = torch.exp(-((f_matrix - F_matrix)**2) / (2 * (sigma**2)))  # Shape: [N, N]
     
     # Normalize each row to sum to 1
     g = g / g.sum(dim=1, keepdim=True)  # Shape: [N, N]
     
     # Move to the same device as X
     return g

def Gauss_smooth_vectorized(X, f, Noct=1, smooth_filter=None):
    """
    based on https://github.com/IoSR-Surrey/MatlabToolbox/blob/4bff1bb2da7c95de0ce2713e7c710a0afa70c705/%2Biosr/%2Bdsp/smoothSpectrum.m
    Smooths the magnitude spectrum X using a Gaussian filter.

    Args:
        X (torch.Tensor): Input spectrum to be smoothed, shape (B, N,).
        f (torch.Tensor): Frequency bins corresponding to the spectrum, shape (N,).
        Noct (int, optional): Number of octaves for smoothing. Default is 1
    Returns:
        torch.Tensor: Smoothed spectrum, same shape as X.
    """
    shape = X.shape
    x_oct = X.clone().view(-1, shape[-1])  # Initialize smoothed output
    
    if Noct > 0:
        # Vectorized implementation
        # Create a matrix of all frequency pairs
        if smooth_filter is None:
            f_matrix = f.unsqueeze(0)  # Shape: [1, N]
            F_matrix = f.unsqueeze(1)  # Shape: [N, 1]
            
            # Calculate sigma for each center frequency
            sigma = (F_matrix / Noct) / np.pi  # Shape: [N, 1]
            
            # Calculate Gaussian weights for all frequency pairs at once
            g = torch.exp(-((f_matrix - F_matrix)**2) / (2 * (sigma**2)))  # Shape: [N, N]
            
            # Normalize each row to sum to 1
            g = g / g.sum(dim=1, keepdim=True)  # Shape: [N, N]
            
            # Move to the same device as X
        else:
            g=smooth_filter

        g = g.to(X.device)
        
        # Apply the filter to each batch element
        # Skip the first bin (i=0) as in the original loop
        x_oct_new = torch.matmul(x_oct, g[1:].T)  # Shape: [B, N-1]
        
        # Keep the first bin unchanged and update the rest
        x_oct[:, 1:] = x_oct_new
        
        # Ensure non-negative values
        x_oct = x_oct.clamp(min=0)
    
    return x_oct.view(shape)  # Reshape back to original dimensions

