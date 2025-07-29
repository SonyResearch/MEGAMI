import os
import math
import sys
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
from fx_model.fx_pipeline import EffectRandomizer

#from fx_model.distribution_presets.clusters_vocals import get_distributions_Cluster0, get_distributions_Cluster1

from fx_model.distribution_presets.clusters_multitrack import get_distributions_Cluster0_vocals, get_distributions_Cluster1_vocals, get_distributions_Cluster0_bass, get_distributions_Cluster1_bass, get_distributions_Cluster0_drums, get_distributions_Cluster1_drums
#from fx_model.distribution_presets.uniform_RMSnorm import get_distributions_uniform
from fx_model.distribution_presets.uniform import get_distributions_uniform

import logging
# Configure at the beginning of your program
logging.basicConfig(level=logging.WARNING)
# Or specifically for numba.cuda.cudadrv.driver
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)  # or logging.ERROR

from utils.collators import collate_multitrack_sim, collate_multitrack_paired

from fx_model.apply_effects_multitrack_utils import simulate_effects

from fx_model.apply_effects_multitrack_utils import multitrack_batched_processing, forward_reshaping

from utils.data_utils import apply_RMS_normalization

# ----------------------------------------------------------------------------



class Taxonomy2Embedding:
    def __init__(self, max_num_classes=3, type="one-hot"):
        """
        input: list of taxonomies, with 4 digits each, dynamically assign codes to one-hot encoding
        """
        assert type=="one-hot", "Only one-hot encoding is supported for now"

        self.num_classes=max_num_classes

        self.dict_taxonomy = {
            "92": torch.tensor([1,0,0]),
            "2": torch.tensor([0,1,0]),
            "11": torch.tensor([0,0,1])
        }

        self.encoding_shape= (max_num_classes,)

        self.counter=0
        self.type=type

    def encode(self, taxonomy_list):  
        
        
        if self.type == "one-hot":
            encoding=torch.zeros((len(taxonomy_list), self.num_classes), dtype=torch.float32)
        
        for i, t in enumerate(taxonomy_list):
            assert t in self.dict_taxonomy.keys()


        return encoding



class Trainer():
    def __init__(self, args=None, dset=None, network=None,  device='cpu', rank=0, world_size=1, distributed=True, tester=None):

        print("HELLO FROM TRAINER") 

        assert args is not None, "args dictionary is None"
        self.args = args

        assert dset is not None, "dset is None"
        self.dset = dset

        assert network is not None, "network is None"
        self.network = network


        assert device is not None, "device is None"
        self.device = device
        
        print("device", device, "rank", rank, "world_size", world_size)

        self.rank = rank
        self.world_size = world_size

        self.tester= tester

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

        self.skip_val=False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

        distribution_uniform = get_distributions_uniform(sample_rate=44100)
        self.effect_randomizer_uniform=EffectRandomizer(sample_rate=44100, distributions_dict=distribution_uniform, device=device)

        distribution_C0_vocals = get_distributions_Cluster0_vocals(sample_rate=44100)
        distribution_C1_vocals = get_distributions_Cluster1_vocals(sample_rate=44100)

        distribution_C0_drums = get_distributions_Cluster0_drums(sample_rate=44100)
        distribution_C1_drums = get_distributions_Cluster1_drums(sample_rate=44100)

        distribution_C0_bass = get_distributions_Cluster0_bass(sample_rate=44100)
        distribution_C1_bass = get_distributions_Cluster1_bass(sample_rate=44100)

        self.taxonomy_ref = {
            "vocals": "92",
            "drums": "11",
            "bass": "2"
        }
        self.taxonomy_ref_inverse = {
            "92": "vocals",
            "11": "drums",
            "2": "bass"
        }

        self.effect_randomizer_C0 = {
            "92": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C0_vocals, device=device),
            "11": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C0_drums, device=device),
            "2": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C0_bass, device=device)
        }
        self.effect_randomizer_C1 = {
            "92": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C1_vocals, device=device),
            "11": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C1_drums, device=device),
            "2": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C1_bass, device=device)
        }
        self.distributions_dicts= {
            "C0": self.effect_randomizer_C0,
            "C1": self.effect_randomizer_C1
        }


        CLAP_args= args.exp.CLAP_args
        assert CLAP_args is not None, "CLAP_args must be provided for CLAP AE"

        original_path = sys.path.copy()
   

        self.losses = {} # This will be used to store the losses for logging

        self.RMS_norm=self.args.exp.RMS_norm  # Use fixed RMS for evaluation, hardcoded for now

        self.use_gated_RMSnorm=self.args.exp.use_gated_RMSnorm
        


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
                #l2 normalize z_af (just in case)
                #z_af = torch.nn.functional.normalize(z_af, dim=-1, p=2)

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)

                z=z* math.sqrt(z.shape[-1]) 
                z_af=z_af* math.sqrt(z_af.shape[-1])



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
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

                #z=z* math.sqrt(z.shape[-1]) 
                #z_af=z_af* math.sqrt(z_af.shape[-1])



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z

            self.FXenc=fxencode_fn
        elif self.args.exp.FXenc_args.type=="fx_encoder2048+AFv5":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
            feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding_v5 import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                z= torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
                z= z* math.sqrt(z.shape[-1])  # rescale to keep the same scale
                #std is approz 1.7
                #normalize to unit variance
                #z=z/1.7

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
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
        elif self.args.exp.FXenc_args.type=="fx_encoder2048+AFv6":
            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus_2048
            feat_extractor = load_fx_encoder_plusplus_2048(Fxencoder_kwargs, self.device)

            from utils.AF_features_embedding_v6 import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z=feat_extractor(x)
                z= torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
                z= z* math.sqrt(z.shape[-1])  # rescale to keep the same scale
                #std is approz 1.7
                #normalize to unit variance
                #z=z/1.7

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
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
                #std is approz 1.7
                #normalize to unit variance
                #z=z/1.7

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
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

        self.loss_weights = None  # This will be used to store the loss weights for each loss function, if needed
        
        if self.args.exp.loss.type == "MSS_mslr":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)

                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0}
            self.loss_fn = loss_fn  

        elif self.args.exp.loss.type == "MSS_mslr+GANv2+fxenc":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            from networks.discriminator import Discriminator
            self.discriminator=Discriminator(
                rates=[1,2,4],
                periods=[2,3,5],
                fft_sizes = [ 512, 256],
            )

            self.discriminator.to(device)

            from utils.GANloss import GANLoss
            self.gan_loss = GANLoss(self.discriminator) 

            self.optimizer_d= torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=1e-4)

            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor_fxenc = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)


            def loss_fn_discriminaror(x_pred, y):
                return self.gan_loss.discriminator_loss(x_pred, y)


            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)


                loss_g, loss_feature = self.gan_loss.generator_loss(x_pred, y)

                z_pred= feat_extractor_fxenc(x_pred)
                z= feat_extractor_fxenc(y)

                # l2 normalize z and z_pred (just in case)
                z_pred = torch.nn.functional.normalize(z_pred, dim=-1, p=2)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)

                #cosine distance between z_pred and z
                loss_fxenc = 1- torch.nn.functional.cosine_similarity(z_pred, z, dim=-1).mean()


                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "adversarial_loss": loss_g,
                    "feature_loss": loss_feature,  # This is the feature matching loss
                    "loss_fxenc": loss_fxenc,
                }



                return loss_dictionary


            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "adversarial_loss": [], "feature_loss": [], "loss_discriminator": [], "loss_fxenc": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "adversarial_loss": 0.15, "feature_loss": 0.05, "loss_discriminator": 0.15, "loss_fxenc": 1.0}
            self.loss_fn = loss_fn  
            self.loss_fn_discriminator = loss_fn_discriminaror

        elif self.args.exp.loss.type == "MSS_mslr+GAN":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            from networks.discriminator import Discriminator
            self.discriminator=Discriminator()
            self.discriminator.to(device)

            from utils.GANloss import GANLoss
            self.gan_loss = GANLoss(self.discriminator) 

            self.optimizer_d= torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=1e-4)


            def loss_fn_discriminaror(x_pred, y):
                return self.gan_loss.discriminator_loss(x_pred, y)


            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)


                loss_g, loss_feature = self.gan_loss.generator_loss(x_pred, y)

                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "adversarial_loss": loss_g,
                    "feature_loss": loss_feature,  # This is the feature matching loss
                }



                return loss_dictionary


            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "adversarial_loss": [], "feature_loss": [], "loss_discriminator": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "adversarial_loss": 0.15, "feature_loss": 0.05, "loss_discriminator": 0.15}
            self.loss_fn = loss_fn  
            self.loss_fn_discriminator = loss_fn_discriminaror

        elif self.args.exp.loss.type == "MSS_mslr+fxenc+SL1":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor_fxenc = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)

            smooth_l1 = torch.nn.SmoothL1Loss().to(self.device)


            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """
                x_pred=x_pred.contiguous()

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)

                z_pred= feat_extractor_fxenc(x_pred)
                z= feat_extractor_fxenc(y)

                # l2 normalize z and z_pred (just in case)
                z_pred = torch.nn.functional.normalize(z_pred, dim=-1, p=2)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)

                #cosine distance between z_pred and z
                loss_fxenc = 1- torch.nn.functional.cosine_similarity(z_pred, z, dim=-1).mean()

                loss_sl1 = smooth_l1(x_pred, y)


                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "loss_fxenc": loss_fxenc,
                    "loss_sl1": loss_sl1
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_fxenc": [], "loss_sl1": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "loss_fxenc": 1.5, "loss_sl1": 100.0}
            self.loss_fn = loss_fn  
        elif self.args.exp.loss.type == "MSS_mslr+fxenc":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor_fxenc = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)


            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)

                z_pred= feat_extractor_fxenc(x_pred)
                z= feat_extractor_fxenc(y)

                # l2 normalize z and z_pred (just in case)
                z_pred = torch.nn.functional.normalize(z_pred, dim=-1, p=2)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)

                #cosine distance between z_pred and z
                loss_fxenc = 1- torch.nn.functional.cosine_similarity(z_pred, z, dim=-1).mean()


                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "loss_fxenc": loss_fxenc,
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_fxenc": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "loss_fxenc": 1.5}
            self.loss_fn = loss_fn  
        elif self.args.exp.loss.type == "MSS_mslr+fxenc+AF":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            Fxencoder_kwargs= self.args.exp.fx_encoder_plusplus_args

            from evaluation.feature_extractors import load_fx_encoder_plusplus
            feat_extractor_fxenc = load_fx_encoder_plusplus(Fxencoder_kwargs, self.device)

            from utils.ITOMaster_loss import compute_log_rms, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)

                z_pred= feat_extractor_fxenc(x_pred)
                z= feat_extractor_fxenc(y)

                # l2 normalize z and z_pred (just in case)
                z_pred = torch.nn.functional.normalize(z_pred, dim=-1, p=2)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)

                #cosine distance between z_pred and z
                loss_fxenc = 1- torch.nn.functional.cosine_similarity(z_pred, z, dim=-1).mean()


                log_rms_pred = compute_log_rms(x_pred)
                log_rms= compute_log_rms(y)

                loss_log_rms = torch.abs(log_rms_pred - log_rms).mean()

                crest_factor_pred = compute_crest_factor(x_pred)
                crest_factor= compute_crest_factor(y)

                loss_crest_factor = torch.abs(crest_factor_pred - crest_factor).mean()

                stereo_width_pred = compute_stereo_width(x_pred)
                stereo_width= compute_stereo_width(y)

                loss_stereo_width = torch.abs(stereo_width_pred - stereo_width).mean()

                stereo_imbalance_pred = compute_stereo_imbalance(x_pred)
                stereo_imbalance= compute_stereo_imbalance(y)
                loss_stereo_imbalance = torch.abs(stereo_imbalance_pred - stereo_imbalance).mean()
                

                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "loss_fxenc": loss_fxenc,
                    "loss_log_rms": loss_log_rms,
                    "loss_crest_factor": loss_crest_factor,
                    "loss_stereo_width": loss_stereo_width,
                    "loss_stereo_imbalance": loss_stereo_imbalance
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_fxenc": [], "loss_log_rms": [], "loss_crest_factor": [], "loss_stereo_width": [], "loss_stereo_imbalance": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "loss_fxenc": 1.0, 
                                 "loss_log_rms": 0.125, "loss_crest_factor": 0.125, 
                                 "loss_stereo_width":1.0 , "loss_stereo_imbalance": 1.00}
            self.loss_fn = loss_fn  

        elif self.args.exp.loss.type == "MSS_mslr+MLDR":

            from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)

            from evaluation.ldr import MLDRLoss
            mldr_lr= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
            ).cuda()
            from evaluation.ldr import MLDRLoss
            mldr_ms= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
                mid_side=True
            ).cuda()

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)


                loss_mldr_midside = mldr_ms(x_pred, y)
                loss_mldr_lr = mldr_lr(x_pred, y)

                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "loss_mldr_midside": loss_mldr_midside,
                    "loss_mldr_lr": loss_mldr_lr
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_mldr_midside": [], "loss_mldr_lr": []}
            self.loss_weights = {"loss_mss_midside": 1.0, "loss_mss_lr": 1.0, "loss_mldr_midside": 0.125, "loss_mldr_lr": 0.25}
            self.loss_fn = loss_fn  


        elif self.args.exp.loss.type == "DiffVox_MSS":
            from auraloss.freq import MultiResolutionSTFTLoss
            mss_lr=MultiResolutionSTFTLoss(
                [128, 512, 2048],
                [32, 128, 512],
                [128, 512, 2048],
                sample_rate=44100,
                perceptual_weighting=True,
            ).cuda()
            from auraloss.freq import  SumAndDifferenceSTFTLoss
            mss_ms=SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
            ).cuda()

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_mss_midside = mss_ms(x_pred, y)
                loss_mss_lr = mss_lr(x_pred, y)

                loss_dictionary = {
                    "loss_mss_midside": 0.5*loss_mss_midside,
                    "loss_mss_lr": loss_mss_lr,
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": []}
            self.loss_fn = loss_fn  
        elif self.args.exp.loss.type == "DiffVox_MSS_MLDR":
            from auraloss.freq import MultiResolutionSTFTLoss
            mss_lr=MultiResolutionSTFTLoss(
                [128, 512, 2048],
                [32, 128, 512],
                [128, 512, 2048],
                sample_rate=44100,
                perceptual_weighting=True,
            ).cuda()
            from auraloss.freq import  SumAndDifferenceSTFTLoss
            mss_ms=SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
            ).cuda()
            from evaluation.ldr import MLDRLoss
            mldr_lr= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
            ).cuda()
            from evaluation.ldr import MLDRLoss
            mldr_ms= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
                mid_side=True
            ).cuda()

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_mss_midside = mss_ms(x_pred, y)
                loss_mss_lr = mss_lr(x_pred, y)

                loss_mldr_midside = mldr_ms(x_pred, y)
                loss_mldr_lr = mldr_lr(x_pred, y)

                loss_dictionary = {
                    "loss_mss_midside": 0.5*loss_mss_midside,
                    "loss_mss_lr": loss_mss_lr,
                    "loss_mldr_midside": 0.25*loss_mldr_midside,
                    "loss_mldr_lr": 0.5*loss_mldr_lr
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_mldr_midside": [], "loss_mldr_lr": []}
            self.loss_fn = loss_fn  

        else:
            raise NotImplementedError("Only MSS_mslr loss is implemented for now")


        if self.args.exp.apply_fxnorm:
            self.fx_normalizer= hydra.utils.instantiate(self.args.exp.fxnorm)

    def apply_random_effects(self, x):

        y=self.effect_randomizer_uniform.forward(x)

        return y

    def simulate_effects(self, x, cluster, taxonomy, masks):
        """
        x: tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        cluster: tensor of shape [B] where B is the batch size
        taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is the tracks, each taxonomy is a string 
        """

        return simulate_effects(x, cluster, taxonomy , 
                                        effect_randomizer_C0=self.effect_randomizer_C0, 
                                        effect_randomizer_C1=self.effect_randomizer_C1, masks=masks)

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

    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        raise ValueError("you should not be here")
        # sigma values are ranged between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max. We need to quantize the values of sigma into 10 logarithmically spaced bins between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max
        torch.nan_to_num(error)  # not tested might crash
        error_mean= error.mean()
        error_mean = error_mean.detach().cpu().numpy()

        training_stats.report('loss', error_mean)

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
                training_stats.report('error_sigma_' + str(self.sigma_bins[i]), error_sigma_i.detach().cpu().numpy())
                #print("training stats")


    def apply_augmentations(self, y, x, augmentations):

        raise NotImplementedError("apply_augmentations is not implemented, please implement it")

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


    #def apply_RMS_normalization(self, x, RMS=None):

    #    if RMS is None:
    #        RMS= torch.tensor(self.RMS_norm, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation
    #    else:
    #        RMS= RMS.view(-1, 1, 1)  

    #    x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

    #    gain= RMS - x_RMS
    #    gain_linear = 10 ** (gain / 20 + 1e-5)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
    #    x=x* gain_linear.view(-1, 1, 1)

    #    return x
    

    def train_step(self):
        '''Training step'''

        self.optimizer.zero_grad()

        a = time.time()

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

            if y.isnan().any():
                print("y has NaN values, skipping step")
                raise ValueError("y has NaN values, skipping step")

            #stereo to mono of x and y
            if x.shape[1] == 2:
                x = x.mean(dim=1, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent

            if y.shape[1] == 1:
                y = y.expand(-1, 2, -1)
            
            #RMS normalization of x and y
            x=apply_RMS_normalization(x, use_gate=self.args.exp.use_gated_RMSnorm)

            z=self.FXenc(y)
            if z.isnan().any():
                print("z has NaN values, skipping step")
                raise ValueError("z has NaN values, skipping step") 
            print("z shape", z.shape)

            if self.args.exp.rms_normalize_y:
                y_norm= apply_RMS_normalization(y,   use_gate=self.args.exp.use_gated_RMSnorm)
            else:
                y_norm = y

            if self.args.exp.apply_fxnorm:
                x=self.fx_normalizer(x, use_gate=self.args.exp.use_gated_RMSnorm)


        y_pred=self.network(x, z)


        if "GAN"in self.args.exp.loss.type:
            # Discriminator step
            self.optimizer_d.zero_grad()

            # Compute the loss for the discriminator
            loss_discriminator = self.loss_fn_discriminator(y_pred, y_norm)

            loss_discriminator= loss_discriminator* self.loss_weights["loss_discriminator"] if self.loss_weights is not None and "loss_discriminator" in self.loss_weights else 1.0


            loss_discriminator.backward()
            
            #apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)

            self.optimizer_d.step()

            self.losses["loss_discriminator"].append( loss_discriminator.mean().detach().cpu().numpy())


        loss_dictionary=self.loss_fn(y_pred, y_norm)

        loss=0
        for key, val in loss_dictionary.items():
            if torch.isnan(val).any():
                raise ValueError("Loss value is NaN, skipping step")
            if self.loss_weights is not None:
                if key in self.loss_weights:
                    val = val * self.loss_weights[key]
            loss += val.mean()  # Take the mean of the loss across the batch

        if not torch.isnan(loss).any():
            self.optimizer.zero_grad()  # Reset gradients for the generator
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            #print("time applying backward", time.time() - a, "seconds")

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

            # Update weights.
            self.optimizer.step()
            #print("time applying optimizer step", time.time() - a, "seconds")
            #print("optimizer step took", time.time() - a, "seconds")

            if self.rank == 0:
                print("iteration", self.it,"loss", loss.item(),  "time", time.time() - a, loss_dictionary)

                #print(time.time() - a, "seconds for step")

                if self.args.logging.log:
                    #Log loss here maybe
                   with torch.no_grad():
                        for key, val in loss_dictionary.items():
                            self.losses[key].append(val.mean().detach().cpu().numpy())
                
                if self.args.logging.log_audio and self.args.logging.log and self.logged_audio_examples < self.args.logging.num_audio_samples_to_log:
                    for b in range(y_pred.shape[0]):  # Log only the first sample in the batch
                            self.log_audio(y_pred[b].unsqueeze(0).detach(), f"pred_wet_train_{self.logged_audio_examples}", it=self.it)  # Just log first sample
                            self.log_audio(y[b].unsqueeze(0).detach(), f"original_wet_train_{self.logged_audio_examples}", it=self.it)  # Just log first sample
                            self.log_audio(x[b].unsqueeze(0).detach(), f"original_dry_{self.logged_audio_examples}", it=self.it)  # Just log first sample
                            self.logged_audio_examples += 1
                #print("time applying loss logging", time.time() - a, "seconds   ")

        else:
            raise ValueError("Loss value is NaN, skipping step")

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

        #self.losses is a list of losses, we want to report the mean  of the losses


        for key, val in self.losses.items():

            loss_mean = np.mean(val) if len(val) > 0 else 0.0
            self.wandb_run.log({"loss_"+key: loss_mean}, step=self.it)
        
        # Reset the losses for the next logging
        self.losses = {key: [] for key in self.losses.keys()}



    def validation_step(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        raise NotImplementedError("validation_step is not implemented, please implement it")
        for key, val_loader in self.val_set_dict.items():

            val_losses= []

            for data in val_loader:

                collated_data = collate_multitrack_sim(data)

                x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                cluster=collated_data['clusters'].to(self.device)  # cluster is a tensor of shape [B, N] where B is the batch size and N is the number of tracks
                taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
                masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

                with torch.no_grad():
                    y=self.simulate_effects(x, cluster, taxonomy, masks)

                    x, _ = forward_reshaping(x, taxonomy, masks=masks)  # reshape x to [B*N, C, L] and taxonomy to [B*N], take into account the masks
                    y, _ = forward_reshaping(y, taxonomy, masks=masks)  # reshape y to [B*N, C, L] and taxonomy to [B*N], take into account the masks

                    #stereo to mono of x and y
                    if x.shape[1] == 2:
                        x = x.mean(dim=1, keepdim=True).expand(-1, 2, -1)  # expand to [B*N, 1, L] to keep the shape consistent
                    
                    #RMS normalization of x and y
                    x= apply_RMS_normalization(x, use_gate=self.args.exp.use_gated_RMSnorm)

                    #rms normalization of y to match the training data.. hardcoded... please change this later

                    z=self.FXenc(y)

                    y= apply_RMS_normalization(y, use_gate=self.args.exp.use_gated_RMSnorm)  # apply RMS normalization to y


                    y_pred=self.network(x, z)

                    val_loss=self.loss_fn(y_pred, y)


                    #replace NaN values with 0
                    val_loss = torch.nan_to_num(val_loss, nan=0.0, posinf=0.0, neginf=0.0)


                    val_loss = val_loss.mean()
                    val_losses.append(val_loss.item())

            # Log the validation loss
            val_loss_mean = np.mean(val_losses)

            self.wandb_run.log({f'val_loss_{key}': val_loss_mean}, step=self.it)

            print(f"Validation {key} - Loss: {val_loss_mean} ")



    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        if self.tester is not None:
            if self.latest_checkpoint is not None:
                print("Loading latest checkpoint", self.latest_checkpoint)
                self.tester.load_checkpoint(self.latest_checkpoint)
            else:
                print("No latest checkpoint found, skipping heavy logging???")
            # setup wandb in tester
            self.tester.do_test(it=self.it)

    def log_audio(self, x, name, it=None):
        if it is None:
            it= self.it
        string = name + "_" + self.args.tester.name

        #dividing by 2 to avoid clipping
        x = x / 2.0


        #print("x shape before logging", x.shape)
        audio_path = utils_logging.write_audio_file(x, self.args.exp.sample_rate, string, path=self.args.model_dir,
                                                    normalize=False, stereo=True)
        self.wandb_run.log({"audio_" + str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},
                           step=it)

    def training_loop(self):

        self.logged_audio_examples=0

        if self.distributed:
            dist.barrier()  

        while True:
            #try:
            a= time.time()
            #try:
            self.train_step()
            #except Exception as e:
            #    print("Error during training step:", e)
            #    print("Skipping this step")
            #    continue
            #print("time for main step", time.time() - a, "seconds")


            if self.rank == 0:
                self.update_ema()


                if self.it > 0 and self.it % self.args.logging.save_interval == 0 and self.args.logging.save_model:
                    self.save_checkpoint()
    
                if self.it > 0 and self.it % self.args.logging.heavy_log_interval == 0 and self.args.logging.log:
                    if self.skip_val:
                        print("Skipping validation")
                        self.skip_val = False
                    else:
                        #self.validation_step()
                        with torch.no_grad():
                            self.heavy_logging()
    
                if self.it > 0 and self.it % self.args.logging.log_interval == 0 and self.args.logging.log:
                    self.easy_logging()

                
                    


            #print("sync", self.rank)
            if self.distributed:
                dist.barrier()  

            # Update state.
            self.it += 1
            #print("time for train step", time.time() - a, "seconds")
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

    # ----------------------------------------------------------------------------
