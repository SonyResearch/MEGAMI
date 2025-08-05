import os
import sys
import json
import hydra
import torch
import numpy as np
import math
import time
import copy
from glob import glob
import re
import wandb
import omegaconf
import utils.log as utils_logging
import utils.training_utils as t_utils
from utils.data_utils import apply_RMS_normalization
import copy
import auraloss
from utils.collators import collate_multitrack_paired

# ----------------------------------------------------------------------------


class Trainer():
    def __init__(self, args=None, dset=None, network=None, val_set=None,  device='cpu'):

        print("HELLO FROM TRAINER") 

        assert args is not None, "args dictionary is None"
        self.args = args

        assert dset is not None, "dset is None"
        self.dset = dset

        assert network is not None, "network is None"
        self.network = network

        self.val_set = val_set


        assert device is not None, "device is None"
        self.device = device
        
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

        self.network.to(self.device)

        # Logger Setup
        if self.args.logging.log:
                self.setup_wandb()

        self.skip_val=False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

        self.losses = {} # This will be used to store the losses for logging

        self.use_gated_RMSnorm=self.args.exp.use_gated_RMSnorm
        
        self.loss_weights = None  # This will be used to store the loss weights for each loss function, if needed
        
        if self.args.exp.loss.type == "sd":

            sd_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
                    fft_sizes=[4096, 1024, 256],
                    hop_sizes=[2048, 512, 128],
                    win_lengths=[4096, 1024, 256],
            )

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """


                loss_sd= sd_loss(x_pred, y)

                loss_dictionary = {
                    "loss_sd": loss_sd,
                }

                return loss_dictionary

            self.losses = {"loss_sd": []}
            self.loss_weights = {"loss_sd": 1.0 }
            self.loss_fn = loss_fn  

        else:
            raise NotImplementedError("Only MSS_mslr loss is implemented for now")


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


    def get_batch(self):
        ''' Get an audio example from dset and apply the transform (spectrogram + compression)'''
        data = next(self.dset)

        #assert len(data)== self.args.exp.batch_size, "Batch size mismatch, expected {}, got {}".format(self.args.exp.batch_size, len(data))
        #collated_data = collate_multitrack_paired(data, max_tracks=self.args.exp.max_tracks)  # collate the data, this will pad the data to the maximum number of tracks in the batch

        #x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        #y=collated_data['y'].to(self.device)  # cluster is a tensor of shape [B, N] where B is the batch size and N is the number of tracks
        #taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
        #masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

        x,y =data

        x=x.to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        y=y.to(self.device)



        return x, y


    def train_step(self):
        '''Training step'''

        self.optimizer.zero_grad()

        a = time.time()

        with torch.no_grad():
            x, y = self.get_batch()

        #shape of x is [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        #shape of y is [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        #shape of masks is [B, N] where B is the batch size and N is the number of tracks

        with torch.no_grad():

            if y.isnan().any():
                print("y has NaN values, skipping step")
                raise ValueError("y has NaN values, skipping step")

            #stereo to mono of x and y
            if x.shape[2] == 2:
                x = x.mean(dim=2, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent
            
            x=x.squeeze(2)  # now shape is [B, N, L]

            y=y.sum(dim=1, keepdim=False)  # sum over the tracks, now shape is [B, C, L]

            if y.shape[1] == 1:
                y = y.expand(-1, 2, -1)
            
            #RMS normalization of x and y
            x_norm= x.view(x.shape[0] * x.shape[1], 1, x.shape[2])  # reshape to [B*N, 1, L]
            x_norm=apply_RMS_normalization(x_norm, use_gate=self.args.exp.use_gated_RMSnorm)
            x=x_norm.view(x.shape[0], x.shape[1], x.shape[2])  # reshape back to [B, N, L]

            #y will be peak normalized (each instance in the batch individually)
            #y=apply_RMS_normalization(y, use_gate=True, stereo=True)  # apply RMS normalization to y, this will normalize the peak of each instance in the batch

            #for i in range(y.shape[0]):
            #    # Find the peak value for this batch instance
            #    peak_value = torch.max(torch.abs(y[i]))
            #    y[i] = y[i] / peak_value

        masks=torch.ones((x.shape[0], x.shape[1]), device=self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

        y_pred,_=self.network(x, masks)

        print("y_pred shape", y_pred.shape, "y shape", y.shape)
        loss_dictionary=self.loss_fn(y_pred, y)

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

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

            # Update weights.
            self.optimizer.step()

            print("iteration", self.it,"loss", loss.item(),  "time", time.time() - a)

            if self.args.logging.log:
                    #Log loss here maybe
                   with torch.no_grad():
                        for key, val in loss_dictionary.items():
                            self.losses[key].append(val.mean().detach().cpu().numpy())
                
            if self.args.logging.log_audio and self.args.logging.log and self.logged_audio_examples < self.args.logging.num_audio_samples_to_log:
                    for b in range(y_pred.shape[0]):  # Log only the first sample in the batch
                            self.log_audio(y_pred[b].unsqueeze(0).detach(), f"pred_wet_train_{self.logged_audio_examples}", it=self.it)  # Just log first sample
                            self.log_audio(y[b].unsqueeze(0).detach(), f"original_wet_train_{self.logged_audio_examples}", it=self.it)  # Just log first sample
                            self.logged_audio_examples += 1
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


    def validation(self):
        """
        Do the validation here. This will be called every 10000 iterations or so
        """

        val_loss= 0.0

        i=0
        j=0

        for data in self.val_set:

            with torch.no_grad():
                #collated_data = collate_multitrack_paired(data,  max_tracks=self.args.exp.max_tracks)
    
                #x=collated_data['x'].to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                #y=collated_data['y'].to(self.device)  # y is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                #taxonomy=collated_data['taxonomies']  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
                #masks=collated_data['masks'].to(self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch
                x, y = data
                x=x.to(self.device)  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                y=y.to(self.device)
                masks=torch.ones((x.shape[0], x.shape[1]), device=self.device)  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch
    
                if y.isnan().any():
                    print("y has NaN values, skipping step")
                    raise ValueError("y has NaN values, skipping step")
    
                #stereo to mono of x and y
                if x.shape[2] == 2:
                    x = x.mean(dim=2, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent
                
                x=x.squeeze(2)  # now shape is [B, N, L]
    
                y=y.sum(dim=1, keepdim=False)  # sum over the tracks, now shape is [B, C, L]
    
                if y.shape[1] == 1:
                    y = y.expand(-1, 2, -1)
                
                #RMS normalization of x and y
                x_norm= x.view(x.shape[0] * x.shape[1], 1, x.shape[2])  # reshape to [B*N, 1, L]
                x_norm=apply_RMS_normalization(x_norm, use_gate=self.args.exp.use_gated_RMSnorm)
                x=x_norm.view(x.shape[0], x.shape[1], x.shape[2])  # reshape back to [B, N, L]
    
                #y will be peak normalized (each instance in the batch individually)
                #y=apply_RMS_normalization(y, use_gate=True, stereo=True)  # apply RMS normalization to y, this will normalize the peak of each instance in the batch

                print("x shape", x.shape)
                y_pred,_=self.network(x, masks)

                loss_dictionary=self.loss_fn(y_pred, y)

                val_loss+=loss_dictionary['loss_sd'].mean().item()  # Take the mean of the loss across the batch
                i+=1
                if self.args.logging.log_audio and self.args.logging.log and j < self.args.logging.num_audio_samples_to_log:
                    for b in range(y_pred.shape[0]):  # Log only the first sample in the batch
                            self.log_audio(y_pred[b].unsqueeze(0).detach(), f"val_pred_wet_train_{j}", it=self.it)  # Just log first sample
                            self.log_audio(y[b].unsqueeze(0).detach(), f"val_original_wet_train_{j}", it=self.it)  # Just log first sample
                            j += 1

        val_loss /= i

        self.wandb_run.log({"val_loss": val_loss}, step=self.it)


    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """

        self.validation()

    def log_audio(self, x, name, it=None):
        if it is None:
            it= self.it
        string = name + "_" 

        #dividing by 2 to avoid clipping
        x = x / 2.0


        #print("x shape before logging", x.shape)
        audio_path = utils_logging.write_audio_file(x, self.args.exp.sample_rate, string, path=self.args.model_dir,
                                                    normalize=False, stereo=True)
        self.wandb_run.log({"audio_" + str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},
                           step=it)

    def training_loop(self):

        self.logged_audio_examples=0

        while True:
            #try:
            a= time.time()
            try:
                self.train_step()
            except Exception as e:
                print("Error during training step, skipping this step")
                print(e)
                continue

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

                
            # Update state.
            self.it += 1
            #print("time for train step", time.time() - a, "seconds")
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

    # ----------------------------------------------------------------------------


def worker_init_fn(worker_id, rank=0):
    st = np.random.get_state()[2]
    seed= st + worker_id + rank*100
    print(f"worker_init_fn {worker_id} rank {rank} st {st} seed {seed}")

    np.random.seed(seed)



def _main(args):

    print(f"Current Working Directory: {os.getcwd()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)


    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train_set = hydra.utils.instantiate(args.dset.train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.exp.batch_size,
                                               num_workers=args.exp.num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn, timeout=0, prefetch_factor=5 )
    train_loader = iter(train_loader)

    val_set_dict = {}


    val_set = hydra.utils.instantiate(args.dset.validation)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size  )
    val_set_dict[args.dset.validation.mode] = val_loader


    # Network
    if args.network._target_ == 'networks.unet_octCQT.UNet_octCQT':
        network = hydra.utils.instantiate(args.network, sample_rate=args.exp.sample_rate, audio_len=args.exp.audio_len,
                                          device=device)  # instantiate

    else:
        network = hydra.utils.instantiate(args.network)  # instantiate in trainer better

    network = network.to(device)


    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    trainer = Trainer( args=args, dset=train_loader, network=network, device=device, val_set=val_loader)  # This works

    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network._target_}')
    print(f'Dataset:    {args.dset.train._target_}')
    print(f'Batch size:              {args.exp.batch_size}')
    print(f'Device:       {device}')
    print()

    trainer.training_loop()


@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
