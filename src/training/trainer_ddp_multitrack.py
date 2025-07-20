import os
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

from utils.collators import collate_multitrack_train

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

        #self.simulate_effects_fn_compiled = torch.compile(self.simulate_effects_fn)

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
        #print("error_mean", error_mean)
        #dist.reduce(error_mean, 0, torch.distributed.ReduceOp.AVG)
        #print("error_mean", error_mean)
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
        collated_data = collate_multitrack_train(data, max_tracks=self.args.exp.max_tracks, sample_rate=self.args.exp.sample_rate, segment_length=self.args.exp.audio_len, device=self.device)  # collate the data, this will pad the data to the maximum number of tracks in the batch

        y=collated_data['y']  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        masks=collated_data['masks']  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

        #print("x shape", x.shape, "cluster shape", cluster.shape, "taxonomy ", taxonomy, "masks shape", masks.shape)

        #print("x rms", 20* torch.log10(x.std(dim=(2, 3)) + 1e-6))


        return y, masks



    def train_step(self):
        '''Training step'''

        self.optimizer.zero_grad()


        y, masks = self.get_batch()

        if torch.isnan(y).any():
            raise ValueError("Input tensor contains NaN values")

        if self.distributed:
            dist.barrier()

        a = time.time()

        error, sigma, x_norm, y = self.diff_params.loss_fn(self.network, sample=y, context=None, ema=self.ema, clusters=None,  masks=masks, compile=self.args.exp.compile)

        loss = error.mean(dim=1)
        loss = loss.mean()

        if not torch.isnan(loss).any():
            loss.backward()

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

            # Update weights.
            self.optimizer.step()

            if self.rank == 0:
                print("iteration", self.it,"loss", loss.item(), "time", time.time() - a)

                if self.args.logging.log:
                    self.process_loss_for_logging(error, sigma)

                    
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

        training_stats.default_collector.update()
        loss_mean = training_stats.default_collector.mean('loss')
        self.wandb_run.log({'loss': loss_mean}, step=self.it)

        # Fancy plot for error with respect to sigma
        sigma_means = []
        sigma_stds = []
        for i in range(len(self.sigma_bins)):
            a = training_stats.default_collector.mean('error_sigma_' + str(self.sigma_bins[i]))
            #replace NaN with 0
            if np.isnan(a):
                a = 0
            a=np.float32(a)  #convert to float32
            #set dtype to float32
            sigma_means.append(a)

            self.wandb_run.log({'error_sigma_' + str(self.sigma_bins[i]): a}, step=self.it)
            a = training_stats.default_collector.std('error_sigma_' + str(self.sigma_bins[i]))
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

        #dividing by 2 to avoid clipping
        x = x / 2.0


        #print("x shape before logging", x.shape)
        audio_path = utils_logging.write_audio_file(x, self.args.exp.sample_rate, string, path=self.args.model_dir,
                                                    normalize=False, stereo=True)
        self.wandb_run.log({"audio_" + str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},
                           step=self.it)

        #if self.args.logging.log_spectrograms:
        #    spec_sample = utils_logging.plot_spectrogram_from_raw_audio(x, self.args.logging.stft)
        #    self.wandb_run.log({"spec_" + str(string): spec_sample}, step=self.it)

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
                # If an error occurs, we skip the step and continue training
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
                        #pass
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
