import os
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

import logging
# Configure at the beginning of your program
logging.basicConfig(level=logging.WARNING)
# Or specifically for numba.cuda.cudadrv.driver
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)  # or logging.ERROR

from utils.collators import collate_multitrack_sim

from fx_model.apply_effects_multitrack_utils import simulate_effects

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
    def __init__(self, args=None, dset=None, val_set_dict=None, network=None,  device='cpu', rank=0, world_size=1, distributed=True):

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

        self.distributed = distributed
        if distributed:
            self.network = torch.nn.parallel.DistributedDataParallel(self.network.to(device), device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)

        if self.args.exp.compile:
            self.network=torch.compile(self.network)

        # Logger Setup
        if self.rank == 0:
            if self.args.logging.log:
                self.setup_wandb()

        self.skip_val=False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

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
   
        from evaluation.feature_extractors import load_CLAP
        CLAP_encoder= load_CLAP(CLAP_args, device=self.device)

        sys.path = original_path

        def encode_fn(x, *args):
            x=x.to(self.device)
            z=CLAP_encoder(x, type="dry") #shape (B, C)
            return z

        self.CLAP_encode=encode_fn

        self.losses = []  # This will be used to store the losses for logging

        self.val_set_dict=val_set_dict

        self.RMS_norm=-25

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

    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
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

    def get_batch(self):
        ''' Get an audio example from dset and apply the transform (spectrogram + compression)'''
        data = next(self.dset)

        assert len(data)== self.args.exp.batch_size, "Batch size mismatch, expected {}, got {}".format(self.args.exp.batch_size, len(data))
        collated_data = collate_multitrack_sim(data, max_tracks=self.args.exp.max_tracks)  # collate the data, this will pad the data to the maximum number of tracks in the batch

        x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        cluster=collated_data['clusters'].to(self.device)  # cluster is a tensor of shape [B, N] where B is the batch size and N is the number of tracks
        taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
        masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

        #print("x shape", x.shape, "cluster shape", cluster.shape, "taxonomy ", taxonomy, "masks shape", masks.shape)

        #print("x rms", 20* torch.log10(x.std(dim=(2, 3)) + 1e-6))


        return x, cluster, taxonomy, masks

    def simulate_effects(self, x, cluster, taxonomy, masks):
        """
        x: tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        cluster: tensor of shape [B] where B is the batch size
        taxonomy: list of lists of taxonomies. Outer list is the batch, inner list is the tracks, each taxonomy is a string 
        """

        return simulate_effects(x, cluster, taxonomy , 
                                        effect_randomizer_C0=self.effect_randomizer_C0, 
                                        effect_randomizer_C1=self.effect_randomizer_C1, masks=masks)


    def apply_RMS_normalization(self, x):

        RMS= torch.tensor(self.RMS_norm, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear.view(-1, 1, 1)

        return x

    def train_step(self):
        '''Training step'''

        self.optimizer.zero_grad()


        x, cluster, taxonomy, masks = self.get_batch()

        #print("get batch took", time.time() - a, "seconds")
        assert masks.sum()==masks.numel()

        if self.distributed:
            dist.barrier()

        a = time.time()

        with torch.no_grad():
            y=self.simulate_effects(x, cluster, taxonomy, masks)
            x=x.view(-1, x.shape[-2], x.shape[-1])  # flatten the batch and the number of tracks, so that we have a tensor of shape [B*N, C, L]
            y=y.view(-1, y.shape[-2], y.shape[-1])  #

            if x.shape[1] == 2:
                x = x.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            elif x.shape[1] == 1:  # if x is mono, we expand it to stereo
                x = x.expand(-1, 2, -1)
            if y.shape[1] == 2:
                y = y.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            elif y.shape[1] == 1:  # if y is mono, we expand it to stereo   
                y = y.expand(-1, 2, -1)

            x= self.apply_RMS_normalization(x)  # apply RMS normalization to x
            y= self.apply_RMS_normalization(y)  # apply RMS normalization to y

            x_encoded = self.CLAP_encode(x)  # encode x with CLAP encoder
            y_encoded = self.CLAP_encode(y)  # encode y with CLAP encoder
        
            cossim_base= torch.nn.functional.cosine_similarity(x_encoded, y_encoded, dim=1)  # cosine similarity between the encoded features

        x_pred=self.network(y_encoded)

        #loss is cosine similarity between x_pred and x_encoded. here coded with pytorch

        cossim=torch.nn.functional.cosine_similarity(x_pred, x_encoded, dim=1)  # cosine similarity between the predicted and the encoded features
        loss = 1 - cossim  # we want to minimize the cosine similarity, so we take 1 - cosine similarity

        loss = loss.mean()

        if not torch.isnan(loss).any():
            loss.backward()

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

            # Update weights.
            self.optimizer.step()
            #print("optimizer step took", time.time() - a, "seconds")

            if self.rank == 0:
                print("iteration", self.it,"cossim", cossim.mean().item(),"cossim_base", cossim_base.mean().item(), "time", time.time() - a)

                if self.args.logging.log:
                    #Log loss here maybe
                    self.losses.append(loss.item())

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

        #self.losses is a list of losses, we want to report the mean  of the losses

        loss_mean = np.mean(self.losses)
        self.losses = []  # reset the losses after logging

        self.wandb_run.log({'loss': loss_mean}, step=self.it)


    def validation_step(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        for key, val_loader in self.val_set_dict.items():

            val_losses= []
            val_cossims = []

            for data in val_loader:

                collated_data = collate_multitrack_sim(data)

                x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                cluster=collated_data['clusters'].to(self.device)  # cluster is a tensor of shape [B, N] where B is the batch size and N is the number of tracks
                taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
                masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

                with torch.no_grad():
                    y=self.simulate_effects(x, cluster, taxonomy, masks)
                    x=x.view(-1, x.shape[-2], x.shape[-1])  # flatten the batch and the number of tracks, so that we have a tensor of shape [B*N, C, L]
                    y=y.view(-1, y.shape[-2], y.shape[-1])  #

                    if x.shape[1] == 2:
                        x = x.mean(dim=1, keepdim=True).expand(-1, 2, -1)
                    elif x.shape[1] == 1:  # if x is mono, we expand it to stereo
                        x = x.expand(-1, 2, -1)
                    if y.shape[1] == 2:
                        y = y.mean(dim=1, keepdim=True).expand(-1, 2, -1)
                    elif y.shape[1] == 1:  # if y is mono, we expand it to stereo   
                        y = y.expand(-1, 2, -1)

                    x= self.apply_RMS_normalization(x)  # apply RMS normalization to x
                    y= self.apply_RMS_normalization(y)  # apply RMS normalization to y

                    x_encoded = self.CLAP_encode(x)  # encode x with CLAP encoder
                    y_encoded = self.CLAP_encode(y)  # encode y with CLAP encoder

                    x_pred=self.network(y_encoded)

                    val_cossim=torch.nn.functional.cosine_similarity(x_pred, x_encoded, dim=1)  # cosine similarity between the predicted and the encoded features
                    val_loss = 1 - val_cossim  # we want to minimize the cosine similarity, so we take 1 - cosine similarity

                    val_loss = val_loss.mean()
                    val_losses.append(val_loss.item())
                    val_cossims.append(val_cossim.mean().item())

            # Log the validation loss
            val_loss_mean = np.mean(val_losses)

            self.wandb_run.log({f'val_loss_{key}': val_loss_mean}, step=self.it)

            val_cossim_mean = np.mean(val_cossims)
            self.wandb_run.log({f'val_cossim_{key}': val_cossim_mean}, step=self.it)

            print(f"Validation {key} - Loss: {val_loss_mean}, Cossim: {val_cossim_mean}")



    def training_loop(self):

        self.logged_audio_examples=0

        if self.distributed:
            dist.barrier()  

        while True:
            try:
                self.train_step()
            except Exception as e:
                print("Error during training step:", e)
                print("Skipping this step")
                continue

            if self.rank == 0:
                self.update_ema()

                if self.it > 0 and self.it % self.args.logging.save_interval == 0 and self.args.logging.save_model:
                    self.save_checkpoint()
    
                if self.it > 0 and self.it % self.args.logging.heavy_log_interval == 0 and self.args.logging.log:
                    if self.skip_val:
                        print("Skipping validation")
                        self.skip_val = False
                    else:
                        self.validation_step()
    
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
