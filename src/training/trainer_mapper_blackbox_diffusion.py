import os
import sys
import math
import time
import copy
import numpy as np
import torch
from glob import glob
import re
import hydra
import wandb
import omegaconf

from utils.torch_utils import training_stats
import utils.log as utils_logging
import utils.training_utils as t_utils

import torch.distributed as dist

# ----------------------------------------------------------------------------


class Trainer():
    def __init__(self, args=None, dset=None, network=None, diff_params=None, tester=None, device='cpu', rank=0, world_size=1, distributed=True):

        print("HELLO FROM TRAINER") 

        assert args is not None, "args dictionary is None"
        self.args = args

        assert dset is not None, "dset is None"
        self.dset = dset

        assert network is not None, "network is None"
        self.network = network

        assert diff_params is not None, "diff_params is None"
        self.diff_params = diff_params

        assert device is not None, "device is None"
        self.device = device
        
        print("device", device, "rank", rank, "world_size", world_size)

        self.rank = rank
        self.world_size = world_size

        self.tester = tester

        self.optimizer = hydra.utils.instantiate(args.exp.optimizer, params=network.parameters())
        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)

        # Torch settings
        torch.manual_seed(self.args.exp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        self.total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        if self.rank == 0:
            print("total_params: ", self.total_params / 1e6, "M")

        # Checkpoint Resuming
        self.latest_checkpoint = None
        resuming = False
        if self.args.exp.resume:
            print("Tryinto to resume")
            if self.args.exp.resume_checkpoint=="None":
                self.args.exp.resume_checkpoint = None
            if self.args.exp.resume_checkpoint is not None:
                print("chckping is not None")
                print("resuming from", self.args.exp.resume_checkpoint)
                resuming = self.resume_from_checkpoint(checkpoint_path=self.args.exp.resume_checkpoint)
            else:
                print("resuming from cejsdat")
                resuming = self.resume_from_checkpoint()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))
        if not resuming:
            self.it = 0
            self.latest_checkpoint = None
            if tester is not None:
                self.tester.it = 0

        self.distributed = distributed
        if distributed:
            self.network = torch.nn.parallel.DistributedDataParallel(self.network.to(device), device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)

        if self.args.exp.compile:
            self.network=torch.compile(self.network)

        # Logger Setup
        if self.rank == 0:
            if self.args.logging.log:
                self.setup_wandb()
                if self.tester is not None:
                    self.tester.setup_wandb_run(self.wandb_run)
                self.setup_logging_variables()

        self.skip_val=False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

        self.RMS_norm=self.args.exp.RMS_norm  # Use fixed RMS for evaluation, hardcoded for now

        if self.args.exp.FXenc_args.type=="AFxRep":

            AFxRep_args= self.args.exp.AFxRep_args
            from evaluation.feature_extractors import load_AFxRep
            AFxRep_encoder= load_AFxRep(AFxRep_args, device=self.device)
        
            def fxencode_fn(x):

                z=AFxRep_encoder(x)
                return z
            
            self.FXenc=fxencode_fn
            #self.FXenc_compiled=torch.compile(fxencode_fn)
        elif self.args.exp.FXenc_args.type=="AFxRep+AF":

            AFxRep_args= self.args.exp.AFxRep_args
            from evaluation.feature_extractors import load_AFxRep
            AFxRep_encoder= load_AFxRep(AFxRep_args, device=self.device)

            from utils.AF_features_embedding import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)
        
            def fxencode_fn(x):

                z=AFxRep_encoder(x)

                z_af,_=AFembedding.encode(x)

                #l2 normalize z and z_af (just in case)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)
                z_af = torch.nn.functional.normalize(z_af, dim=-1, p=2)

                #rescale z and z_af with sqrt(dim) to keep the same scale
                z=z* math.sqrt(z.shape[-1])
                z_af=z_af* math.sqrt(z_af.shape[-1])


                x_all= torch.cat([z, z_af], dim=-1)
                #now L2 normalize

                return torch.nn.functional.normalize(x_all, dim=-1, p=2)
            
            self.FXenc=fxencode_fn
            #self.FXenc_compiled=torch.compile(fxencode_fn)
        elif self.args.exp.FXenc_args.type=="fx_encoder_++":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                return z
            self.FXenc=fxencode_fn
        elif self.args.exp.FXenc_args.type=="fx_encoder+AFv2":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding_v2 import AF_fourier_embedding

            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                #l2 normalize z (just in case)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)
                z_af,_=AFembedding.encode(x)

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)

                z=z* math.sqrt(z.shape[-1]) 
                z_af=z_af* math.sqrt(z_af.shape[-1])



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                return z_all/ math.sqrt(z_all.shape[-1])  # L2 normalize by dividing by sqrt(dim) to keep the same scale

            self.FXenc=fxencode_fn
        elif self.args.exp.FXenc_args.type=="AFxRep+AFv2":
            
            AFxRep_args= self.args.exp.AFxRep_args
            from evaluation.feature_extractors import load_AFxRep
            AFxRep_encoder= load_AFxRep(AFxRep_args, device=self.device)

            from utils.AF_features_embedding_v2 import AF_fourier_embedding

            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                z=AFxRep_encoder(x)

                z_af,_=AFembedding.encode(x)

                #l2 normalize z and z_af (just in case)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)
                z_af = torch.nn.functional.normalize(z_af, dim=-1, p=2)

                #rescale z and z_af with sqrt(dim) to keep the same scale
                z=z* math.sqrt(z.shape[-1])
                z_af=z_af* math.sqrt(z_af.shape[-1])


                x_all= torch.cat([z, z_af], dim=-1)
                #now L2 normalize

                return torch.nn.functional.normalize(x_all, dim=-1, p=2)

            self.FXenc=fxencode_fn

        elif self.args.exp.FXenc_args.type=="fx_encoder2048+AFv2":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
            feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding_v2 import AF_fourier_embedding

            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                #std is approz 1.7
                #normalize to unit variance
                z=z/1.7

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)


                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z

            self.FXenc=fxencode_fn
        elif self.args.exp.FXenc_args.type=="fx_encoder2048+AFv3":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
            feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding_v2 import AF_fourier_embedding

            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                z= torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
                z= z* math.sqrt(z.shape[-1])  # rescale to keep the same scale

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)

                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize
                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z
            
            def get_log_rms_from_z(z):

                z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale
                AF=z[...,2048:]  # assuming the AF features are the last 2048 dimensions
                AF=AF/ math.sqrt(AF.shape[-1])  # normalize to unit variance

                features= AFembedding.decode(AF)
                log_rms=features[0]

                return log_rms

            self.FXenc=fxencode_fn
            self.get_log_rms_from_z=get_log_rms_from_z  

        elif self.args.exp.FXenc_args.type=="fx_encoder+AF":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding import AF_fourier_embedding

            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                #l2 normalize z (just in case)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)
                z_af,_=AFembedding.encode(x)
                #l2 normalize z_af (just in case)
                z_af = torch.nn.functional.normalize(z_af, dim=-1, p=2)

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)

                z=z* math.sqrt(z.shape[-1]) 
                z_af=z_af* math.sqrt(z_af.shape[-1])



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                return torch.nn.functional.normalize(z_all, dim=-1, p=2)

            self.FXenc=fxencode_fn
        elif self.args.exp.FXenc_args.type=="AF":
            from utils.AF_features_embedding import AF_fourier_embedding

            embedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=embedding.encode(x)
                return z
            
            self.FXenc=fxencode_fn

        else:
            raise NotImplementedError("Only AFxRep is implemented for now")
        

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"] = self.total_params
        self.wandb_run = wandb.init(project=self.args.logging.wandb.project, config=config, dir=self.args.model_dir)
        wandb.watch(self.network, log="all",
                    log_freq=self.args.logging.heavy_log_interval)  # wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name = os.path.basename(
            self.args.model_dir) + "_" + self.args.exp.exp_name + "_" + self.wandb_run.id  # adding the experiment number to the run name, bery important, I hope this does not crash

    def setup_logging_variables(self):
        if self.diff_params.type == "FM":
            self.sigma_bins = np.linspace(self.args.diff_params.sde_hp.t_min, self.args.diff_params.sde_hp.t_max,
                                          num=self.args.logging.num_sigma_bins)
        elif self.diff_params.type == "ve_karras":
            self.sigma_bins = np.logspace(np.log10(self.args.diff_params.sde_hp.sigma_min),
                                          np.log10(self.args.diff_params.sde_hp.sigma_max),
                                          num=self.args.logging.num_sigma_bins, base=10)

    def load_state_dict(self, state_dict):
        return t_utils.load_state_dict(state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer)

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        # Resume training from latest checkpoint available in the output director
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it = 157007  # large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)

            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it = 0

            try:
                checkpoint = torch.load(os.path.join(self.args.model_dir, checkpoint_path), map_location=self.device, weights_only=False)
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it = 157007  # large number to mean that we loaded somethin, but it is arbitrary
                self.network.load_state_dict(checkpoint['ema_model'])
                return True
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it = 0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                print("model_dir", self.args.model_dir)
                print("exp_name", self.args.exp.exp_name)
                if checkpoint_id is None:
                    # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [int(id_regex.search(weight_path).groups()[0])
                                for weight_path in list_weights]
                    checkpoint_id = max(list_ids)

                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device, weights_only=False)
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it = 159000  # large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        if self.distributed:
            if self.args.exp.compile:
                return {
                    'it': self.it,
                    'network': self.network._orig_mod.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema': self.ema.state_dict(),
                    'args': self.args,
                }
            else:
                return {
                    'it': self.it,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema': self.ema.state_dict(),
                    'args': self.args,
                }
        else:
            if self.args.exp.compile:
                return {
                    'it': self.it,
                    'network': self.network._orig_mod.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema': self.ema.state_dict(),
                    'args': self.args,
                }
            else:
                return {
                    'it': self.it,
                    'network': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema': self.ema.state_dict(),
                    'args': self.args,
                }

    def save_checkpoint(self):
        save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
        save_name = f"{self.args.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print("saving", save_name)
        if self.args.logging.remove_old_checkpoints:
            try:
                os.remove(self.latest_checkpoint)
                print("removed last checkpoint", self.latest_checkpoint)
            except:
                print("could not remove last checkpoint", self.latest_checkpoint)
        self.latest_checkpoint = save_name

    def process_loss_for_logging(self, error_dict, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        # sigma values are ranged between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max. We need to quantize the values of sigma into 10 logarithmically spaced bins between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max

        for key, error in error_dict.items():
            torch.nan_to_num(error)  # not tested might crash
            error_mean= error.mean()
            #print("error_mean", error_mean)
            #dist.reduce(error_mean, 0, torch.distributed.ReduceOp.AVG)
            #print("error_mean", error_mean)
            error_mean = error_mean.detach().cpu().numpy()
    
            training_stats.report('loss_'+key, error_mean)
    
            for i in range(len(self.sigma_bins)):
                if i == 0:
                    mask = sigma <= self.sigma_bins[i]
                elif i == len(self.sigma_bins) - 1:
                    mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i - 1])
    
                else:
                    mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i - 1])
                mask = mask.squeeze(-1).cpu()
                if mask.sum() > 0:
                    # find the index of the first element of the mask
                    #idx = np.where(mask == True)[0][0]
    
                    idx=torch.where(mask)[0]
    
                    error_sigma_i=error[idx].mean()
                    #print("error_sigma_i", error_sigma_i)
                    #dist.reduce(error_sigma_i, 0, torch.distributed.ReduceOp.AVG)
                    #print("error_sigma_i", error_sigma_i)
                    training_stats.report('error_sigma_'+key+"_" + str(self.sigma_bins[i]), error_sigma_i.detach().cpu().numpy())
                    #print("training stats")


    def apply_augmentations(self, y, x, augmentations):

        list = augmentations.list
        for a in list:
            if a == "polarity":
                prob = augmentations.polarity.prob
                apply = torch.rand(x.shape[0]) > prob
                y[apply] *= -1
                x[apply] *= -1
            

            else:
                print("augmentation not implemented: " + a)

        return y, x

    def get_batch_paired(self):
        ''' Get an audio example from dset and apply the transform (spectrogram + compression)'''
        x, y = next(self.dset)
        x=x.to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        y=y.to(self.device)

        if y.shape[-1] > self.args.exp.audio_len:
            y = y[:, :, :self.args.exp.audio_len]
        elif y.shape[-1] < self.args.exp.audio_len:
            raise ValueError("y shape is not correct, expected length {}, got {}".format(self.args.exp.audio_len, y.shape[-1]))
        
        if x.shape[-1] > self.args.exp.audio_len:
            x = x[:, :, :self.args.exp.audio_len]
        elif x.shape[-1] < self.args.exp.audio_len:
            raise ValueError("x shape is not correct, expected length {}, got {}".format(self.args.exp.audio_len, x.shape[-1]))

        return x, y

    def get_batch(self):
        ''' Get an audio example from dset and apply the transform (spectrogram + compression)'''
        x = next(self.dset)
        #print("x", x)

        #assert len(data)== self.args.exp.batch_size, "Batch size mismatch, expected {}, got {}".format(self.args.exp.batch_size, len(data))
        #collated_data = collate_multitrack_paired(data, max_tracks=self.args.exp.max_tracks)  # collate the data, this will pad the data to the maximum number of tracks in the batch

        #x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        ##y=collated_data['y'].to(self.device)  # cluster is a tensor of shape [B, N] where B is the batch size and N is the number of tracks
        #taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
        #masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch
        x=x.to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio

        #print("x shape", x.shape, "cluster shape", cluster.shape, "taxonomy ", taxonomy, "masks shape", masks.shape)

        #print("x rms", 20* torch.log10(x.std(dim=(2, 3)) + 1e-6))
        if x.shape[-1] > self.args.exp.audio_len:
            x = x[:, :, :self.args.exp.audio_len]
        elif x.shape[-1] < self.args.exp.audio_len:
            raise ValueError("x shape is not correct, expected length {}, got {}".format(self.args.exp.audio_len, x.shape[-1]))


        return x


    
    def apply_RMS_normalization(self, x, RMS=None):

        if RMS is None:
            RMS= torch.tensor(self.RMS_norm, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation
        else:
            RMS= RMS.view(-1, 1, 1)  

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-5)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear.view(-1, 1, 1)

        return x

    def train_step(self):
        '''Training step'''
        a = time.time()

        self.optimizer.zero_grad()

        with torch.no_grad():
            #x, taxonomy, masks = self.get_batch()
            if self.args.exp.data_type=="diffvox_random":
                x = self.get_batch()
            elif self.args.exp.data_type=="paired":
                x, y = self.get_batch_paired()


        if self.distributed:
            dist.barrier()

        with torch.no_grad():

            if self.args.exp.data_type=="diffvox_random":
                y=self.apply_random_effects(x)  # apply random effects to x

            #stereo to mono of x and y
            if x.shape[1] == 2:
                x = x.mean(dim=1, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent

            if y.shape[1] == 1:
                y = y.expand(-1, 2, -1)
            
            #RMS normalization of x and y
            x= self.apply_RMS_normalization(x)  # apply RMS normalization to x

            z=self.FXenc(y)
            if z.isnan().any():
                print("z has NaN values, skipping step")
                raise ValueError("z has NaN values, skipping step") 
            #print("z shape", z.shape)

            if self.args.exp.rms_normalize_y:
                y= self.apply_RMS_normalization(y)  # apply RMS normalization to y

        error_dict, sigma = self.diff_params.loss_fn(self.network, sample=y, x=x, embedding=z, ema=self.ema)


        loss=0
        for key, value in error_dict.items():
            if torch.isnan(value).any():
                print(f"Error {key} has NaN values, skipping step")
                raise ValueError(f"Error {key} has NaN values, skipping step")
            
            #print(f"Error {key} has shape {value.shape} and mean {value.mean()}")

            loss+=value.mean()

        self.error_dict = error_dict

        if not torch.isnan(loss).any():
            loss.backward()

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

            # Update weights.
            self.optimizer.step()

            if self.rank == 0:
                print("iteration", self.it,"loss", loss.item(), "time", time.time() - a)

                if self.args.logging.log:
                    self.process_loss_for_logging(error_dict, sigma)
        else:
            print("loss is NaN, skipping step")

    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = self.args.exp.ema_rampup  # ema_rampup should be set to 10000 in the config file
        ema_rate = self.args.exp.ema_rate  # ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.exp.batch_size
        with torch.no_grad():
            if self.args.exp.compile:
                # Here self.network is a torch.compile object, so we need to use the original network
                source_params = self.network._orig_mod.parameters()  # Access the original module's parameters
            else:
                source_params = self.network.parameters()

            # Apply EMA update
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), source_params):
                    dst.copy_(dst * s + src * (1 - s))
            else:
                for dst, src in zip(self.ema.parameters(), source_params):
                    dst.copy_(dst * ema_rate + src * (1 - ema_rate))


    def easy_logging(self):
        """
         Do the simplest logging here. This will be called every 1000 iterations or so
        I will use the training_stats.report function for this, and aim to report the means and stds of the losses in wandb
        """

        for key in self.error_dict.keys():

            training_stats.default_collector.update()
            loss_mean = training_stats.default_collector.mean('loss_' + key)
            self.wandb_run.log({'loss_'+key: loss_mean}, step=self.it)
    
            # Fancy plot for error with respect to sigma
            sigma_means = []
            sigma_stds = []
            for i in range(len(self.sigma_bins)):
                a = training_stats.default_collector.mean('error_sigma_'+key+'_' + str(self.sigma_bins[i]))
                #replace NaN with 0
                if np.isnan(a):
                    a = 0
                a=np.float32(a)  #convert to float32
                #set dtype to float32
                sigma_means.append(a)
    
                self.wandb_run.log({'error_sigma_' +key+'_'+ str(self.sigma_bins[i]): a}, step=self.it)
                a = training_stats.default_collector.std('error_sigma_' +key+'_'+ str(self.sigma_bins[i]))
                if np.isnan(a):
                    a = 0
                a=np.float32(a)  #convert to float32
                sigma_stds.append(a)

        #figure = utils_logging.plot_loss_by_sigma(sigma_means, sigma_stds, self.sigma_bins,
        #                                          log_scale=False if self.diff_params.type == "FM" else True)
        #wandb.log({"loss_dependent_on_sigma": figure}, step=self.it, commit=True)

    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        if self.tester is not None:
            if self.latest_checkpoint is not None:
                self.tester.load_checkpoint(self.latest_checkpoint)
            # setup wandb in tester
            self.tester.do_test(it=self.it)

    def log_audio(self, x, name):
        string = name + "_" + self.args.tester.name

        audio_path = utils_logging.write_audio_file(x, self.args.exp.sample_rate, string, path=self.args.model_dir,
                                                    normalize=False, stereo=True)
        self.wandb_run.log({"audio_" + str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},
                           step=self.it)

        #if self.args.logging.log_spectrograms:
        #    spec_sample = utils_logging.plot_spectrogram_from_raw_audio(x, self.args.logging.stft)
        #    self.wandb_run.log({"spec_" + str(string): spec_sample}, step=self.it)

    def training_loop(self):

        if self.distributed:
            dist.barrier()  

        while True:
            self.train_step()

            if self.rank == 0:
                self.update_ema()

                if self.it > 0 and self.it % self.args.logging.save_interval == 0 and self.args.logging.save_model:
                    self.save_checkpoint()
    
                if self.it > 0 and self.it % self.args.logging.heavy_log_interval == 0 and self.args.logging.log:
                    if self.skip_val:
                        print("Skipping validation")
                        self.skip_val = False
                    else:
                        self.heavy_logging()
    
                if self.it > 0 and self.it % self.args.logging.log_interval == 0 and self.args.logging.log:
                    self.easy_logging()


            #print("sync", self.rank)
            if self.distributed:
                dist.barrier()  

            # Update state.
            self.it += 1
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

    # ----------------------------------------------------------------------------
