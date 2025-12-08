# Copyright (c) 2025 Sony Research
# Licensed under CC BY-NC-SA 4.0
# See LICENSE file for details

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
from inference.validator_FxProcessor import ValidatorFxProcessor
import copy

# ----------------------------------------------------------------------------


class FxProcessorTrainer:
    def __init__(self, args=None, dset=None, network=None, device="cpu", tester=None):
        """
        Trainer class for training FxProcessor models.

        Args:
            args (omegaconf.DictConfig): Configuration dictionary.
            dset (iterable): Dataset iterator for training data.
            network (torch.nn.Module): Neural network model to be trained.
            device (str or torch.device): Device to run the training on.
            tester (ValidatorFxProcessor, optional): Validator for evaluating the model during training.

        """

        assert args is not None, "args dictionary is None"
        self.args = args

        assert dset is not None, "dset is None"
        self.dset = dset

        assert network is not None, "network is None"
        self.network = network

        assert device is not None, "device is None"
        self.device = device

        self.tester = tester

        self.optimizer = hydra.utils.instantiate(
            args.exp.optimizer, params=network.parameters()
        )
        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)

        # Torch settings
        torch.manual_seed(self.args.exp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        self.total_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )

        # Checkpoint Resuming
        self.latest_checkpoint = None
        resuming = False
        if self.args.exp.resume:
            if self.args.exp.resume_checkpoint == "None":
                self.args.exp.resume_checkpoint = None
            if self.args.exp.resume_checkpoint is not None:
                print("resuming from", self.args.exp.resume_checkpoint)
                resuming = self.resume_from_checkpoint(
                    checkpoint_path=self.args.exp.resume_checkpoint
                )
            else:
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

        self.network.to(self.device)

        # Logger Setup
        if self.args.logging.log:
            self.setup_wandb()
            if self.tester is not None:
                self.tester.setup_wandb_run(self.wandb_run)

        self.skip_val = False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

        self.losses = {}  # This will be used to store the losses for logging

        self.use_gated_RMSnorm = self.args.exp.use_gated_RMSnorm

        if self.args.exp.style_encoder_type == "FxEncoder++_DynamicFeatures":

            Fxencoder_kwargs = self.args.exp.fx_encoder_plusplus_args

            from utils.feature_extractors.load_features import (
                load_fx_encoder_plusplus_2048,
            )

            feat_extractor = load_fx_encoder_plusplus_2048(
                Fxencoder_kwargs, self.device
            )

            from utils.feature_extractors.AF_features_embedding import (
                AF_fourier_embedding,
            )

            AFembedding = AF_fourier_embedding(device=self.device)

            def fxencode_fn(x):
                """
                x: tensor of shape [B, C, L] where B is the batch size, C is the number of channels and L is the length of the audio
                """
                z = feat_extractor(x)
                z = torch.nn.functional.normalize(
                    z, dim=-1, p=2
                )  # normalize to unit variance
                z = z * math.sqrt(z.shape[-1])  # rescale to keep the same scale

                z_af, _ = AFembedding.encode(x)
                # embedding is l2 normalized, normalize to unit variance
                z_af = z_af * math.sqrt(
                    z_af.shape[-1]
                )  # rescale to keep the same scale

                # concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)
                z_all = torch.cat([z, z_af], dim=-1)

                norm_z = z_all / math.sqrt(
                    z_all.shape[-1]
                )  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z

            self.style_encode = fxencode_fn
        else:
            raise NotImplementedError(
                "Only FxEncoder++_DynamicFeatures is implemented for now"
            )

        self.loss_weights = None  # This will be used to store the loss weights for each loss function, if needed

        if self.args.exp.loss_type == "MSS_mslr+fxenc":

            from utils.MSS_loss import MultiScale_Spectral_Loss_MidSide_DDSP

            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(
                mode="midside", eps=1e-6, device=device
            )
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(
                mode="ori", eps=1e-6, device=device
            )

            Fxencoder_kwargs = self.args.exp.fx_encoder_plusplus_args

            from utils.feature_extractors.load_features import load_fx_encoder_plusplus

            feat_extractor_fxenc = load_fx_encoder_plusplus(
                Fxencoder_kwargs, self.device
            )

            def loss_fn(x_pred, y):
                """
                Loss function, which is the mean squared error between the denoised latent and the clean latent
                Args:
                    x_pred (Tensor): shape: (B,T) Intermediate noisy latent to denoise
                    y (Tensor): shape: (B,T) Clean latent
                """

                loss_midside = multi_scale_spectral_midside(x_pred, y)
                loss_ori = multi_scale_spectral_ori(x_pred, y)

                z_pred = feat_extractor_fxenc(x_pred)
                z = feat_extractor_fxenc(y)

                # l2 normalize z and z_pred (just in case)
                z_pred = torch.nn.functional.normalize(z_pred, dim=-1, p=2)
                z = torch.nn.functional.normalize(z, dim=-1, p=2)

                # cosine distance between z_pred and z
                loss_fxenc = (
                    1 - torch.nn.functional.cosine_similarity(z_pred, z, dim=-1).mean()
                )

                loss_dictionary = {
                    "loss_mss_midside": loss_midside,
                    "loss_mss_lr": loss_ori,
                    "loss_fxenc": loss_fxenc,
                }

                return loss_dictionary

            self.losses = {"loss_mss_midside": [], "loss_mss_lr": [], "loss_fxenc": []}
            self.loss_weights = {
                "loss_mss_midside": 1.0,
                "loss_mss_lr": 1.0,
                "loss_fxenc": 1.5,
            }
            self.loss_fn = loss_fn

        else:
            raise NotImplementedError("Only MSS_mslr loss is implemented for now")

        if self.args.exp.apply_fxnorm:
            self.fx_normalizer = hydra.utils.instantiate(self.args.exp.fxnorm)

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"] = self.total_params
        self.wandb_run = wandb.init(
            project=self.args.logging.wandb.project,
            config=config,
            dir=self.args.model_dir,
        )
        wandb.watch(
            self.network, log="all", log_freq=self.args.logging.heavy_log_interval
        )  # wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name = (
            os.path.basename(self.args.model_dir)
            + "_"
            + self.args.exp.exp_name
            + "_"
            + self.wandb_run.id
        )  # adding the experiment number to the run name, bery important, I hope this does not crash

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the network, ema, and optimizer.
        """
        return t_utils.load_state_dict(
            state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer
        )

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. If None, will search for latest checkpoint.
            checkpoint_id (int, optional): Checkpoint ID to load if checkpoint_path is None.

        Returns:
            bool: True if resuming from checkpoint was successful, False otherwise.

        """
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint["it"]
                except:
                    self.it = 157007  # large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)

            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it = 0

            try:
                checkpoint = torch.load(
                    os.path.join(self.args.model_dir, checkpoint_path),
                    map_location=self.device,
                    weights_only=False,
                )
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint["it"]
                except:
                    self.it = 157007  # large number to mean that we loaded somethin, but it is arbitrary
                self.network.load_state_dict(checkpoint["ema_model"])
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
                    list_ids = [
                        int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights
                    ]
                    checkpoint_id = max(list_ids)

                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt",
                    map_location=self.device,
                    weights_only=False,
                )
                # if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint["it"]
                except:
                    self.it = 159000  # large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        """
        Return the state dictionary containing the current iteration, network state, optimizer state, ema state, and args.
        """
        return {
            "it": self.it,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "args": self.args,
        }

    def save_checkpoint(self):
        """
        Save the current state of the trainer to a checkpoint file.
        """

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
        """
        Get a batch of data from the dataset iterator.
        Check that the length of x and y are correct.

        Returns:
            x (torch.Tensor): Input tensor of shape [B, N, C, L].
            y (torch.Tensor): Target tensor of shape [B, N, C, L].
        """
        x, y = next(self.dset)
        x = x.to(
            self.device
        )  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        y = y.to(self.device)

        if y.shape[-1] > self.args.exp.audio_len:
            y = y[:, :, : self.args.exp.audio_len]
        elif y.shape[-1] < self.args.exp.audio_len:
            raise ValueError(
                "y shape is not correct, expected length {}, got {}".format(
                    self.args.exp.audio_len, y.shape[-1]
                )
            )

        if x.shape[-1] > self.args.exp.audio_len:
            x = x[:, :, : self.args.exp.audio_len]
        elif x.shape[-1] < self.args.exp.audio_len:
            raise ValueError(
                "x shape is not correct, expected length {}, got {}".format(
                    self.args.exp.audio_len, x.shape[-1]
                )
            )

        return x, y

    def train_step(self):
        """
        Perform a single training step.
        """

        self.optimizer.zero_grad()

        a = time.time()

        with torch.no_grad():
            x, y = self.get_batch()

        with torch.no_grad():

            if y.isnan().any():
                raise ValueError("y has NaN values, skipping step")

            # stereo to mono of x and y
            if x.shape[1] == 2:
                x = x.mean(
                    dim=1, keepdim=True
                )  # expand to [B*N, 1, L] to keep the shape consistent

            if y.shape[1] == 1:
                y = y.expand(-1, 2, -1)

            z = self.style_encode(y)
            if z.isnan().any():
                raise ValueError("z has NaN values, skipping step")

            if self.args.exp.rms_normalize_y:
                y_norm = apply_RMS_normalization(
                    y, self.args.exp.RMS_norm, use_gate=self.args.exp.use_gated_RMSnorm
                )  # Apply RMS normalization to the target y
            else:
                y_norm = y

            if self.args.exp.apply_fxnorm:
                x_fxnorm = self.fx_normalizer(
                    x.clone(),
                    use_gate=self.args.exp.use_gated_RMSnorm,
                    RMS=self.args.exp.RMS_norm,
                )
            else:
                x = apply_RMS_normalization(
                    x, self.args.exp.RMS_norm, use_gate=self.args.exp.use_gated_RMSnorm
                )  # Apply RMS normalization to the input x
                x_fxnorm = x.clone()

        y_pred = self.network(x_fxnorm, z)

        loss_dictionary = self.loss_fn(y_pred, y_norm)

        loss = 0
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
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.args.exp.max_grad_norm
                )

            # Update weights.
            self.optimizer.step()

            print(
                "iteration",
                self.it,
                "loss",
                loss.item(),
                "time",
                time.time() - a,
                loss_dictionary,
            )

            if self.args.logging.log:
                # Log loss here maybe
                with torch.no_grad():
                    for key, val in loss_dictionary.items():
                        self.losses[key].append(val.mean().detach().cpu().numpy())

            if (
                self.args.logging.log_audio
                and self.args.logging.log
                and self.logged_audio_examples
                < self.args.logging.num_audio_samples_to_log
            ):
                for b in range(
                    y_pred.shape[0]
                ):  # Log only the first sample in the batch
                    self.log_audio(
                        y_pred[b].unsqueeze(0).detach(),
                        f"aa_pred_wet_train_{self.logged_audio_examples}",
                        it=self.it,
                    )  # Just log first sample
                    self.log_audio(
                        y_norm[b].unsqueeze(0).detach(),
                        f"aa_original_wet_train_{self.logged_audio_examples}",
                        it=self.it,
                    )  # Just log first sample
                    self.log_audio(
                        x[b].unsqueeze(0).detach(),
                        f"aa_original_dry_{self.logged_audio_examples}",
                        it=self.it,
                    )  # Just log first sample
                    self.log_audio(
                        x_fxnorm[b].unsqueeze(0).detach(),
                        f"aa_original_dry_fxnorm_{self.logged_audio_examples}",
                        it=self.it,
                    )  # Just log first sample
                    self.logged_audio_examples += 1
        else:
            raise ValueError("Loss value is NaN, skipping step")

    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = (
            self.args.exp.ema_rampup
        )  # ema_rampup should be set to 10000 in the config file
        ema_rate = (
            self.args.exp.ema_rate
        )  # ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.exp.batch_size
        with torch.no_grad():
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
        """

        # self.losses is a list of losses, we want to report the mean  of the losses

        for key, val in self.losses.items():

            loss_mean = np.mean(val) if len(val) > 0 else 0.0
            self.wandb_run.log({"loss_" + key: loss_mean}, step=self.it)

        # Reset the losses for the next logging
        self.losses = {key: [] for key in self.losses.keys()}

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
        """
        Log audio to wandb.
        """
        if it is None:
            it = self.it
        string = name + "_" + self.args.tester.name

        audio_path = utils_logging.write_audio_file(
            x,
            self.args.exp.sample_rate,
            string,
            path=self.args.model_dir,
            normalize=False,
            stereo=True,
        )
        self.wandb_run.log(
            {
                "audio_"
                + str(string): wandb.Audio(
                    audio_path, sample_rate=self.args.exp.sample_rate
                )
            },
            step=it,
        )

    def training_loop(self):
        """
        Main training loop.
        """

        self.logged_audio_examples = 0

        while True:
            a = time.time()
            try:
                self.train_step()
            except Exception as e:
                print("Error during training step:", e)
                print("Skipping this step")
                continue

            self.update_ema()

            if (
                self.it > 0
                and self.it % self.args.logging.save_interval == 0
                and self.args.logging.save_model
            ):
                self.save_checkpoint()

            if (
                self.it > 0
                and self.it % self.args.logging.heavy_log_interval == 0
                and self.args.logging.log
            ):
                if self.skip_val:
                    print("Skipping validation")
                    self.skip_val = False
                else:
                    with torch.no_grad():
                        self.heavy_logging()

            if (
                self.it > 0
                and self.it % self.args.logging.log_interval == 0
                and self.args.logging.log
            ):
                self.easy_logging()

            self.it += 1
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

    # ----------------------------------------------------------------------------


def worker_init_fn(worker_id, rank=0):
    st = np.random.get_state()[2]
    seed = st + worker_id + rank * 100
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
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.exp.batch_size,
        num_workers=args.exp.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        timeout=0,
        prefetch_factor=5,
    )
    train_loader = iter(train_loader)

    val_set_dict = {}

    val_set = hydra.utils.instantiate(args.dset.validation)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=args.exp.val_batch_size
    )
    val_set_dict[args.dset.validation.mode] = val_loader

    val_set_2 = hydra.utils.instantiate(args.dset.validation_2)
    val_loader_2 = torch.utils.data.DataLoader(
        dataset=val_set_2, batch_size=args.exp.val_batch_size
    )
    val_set_dict[args.dset.validation_2.mode] = val_loader_2

    # Network
    if args.network._target_ == "networks.unet_octCQT.UNet_octCQT":
        network = hydra.utils.instantiate(
            args.network,
            sample_rate=args.exp.sample_rate,
            audio_len=args.exp.audio_len,
            device=device,
        )  # instantiate
    else:
        network = hydra.utils.instantiate(args.network)  # instantiate in trainer better

    network = network.to(device)

    args.tester.sampling_params.same_as_training = True  # Make sure that we use the same HP for sampling as the ones used in training
    args.tester.wandb.use = False  # Will do that in training
    network_tester = copy.deepcopy(network).eval().requires_grad_(False)
    tester = ValidatorFxProcessor(
        args,
        network_tester,
        device=device,
        in_training=True,
        test_set_dict=val_set_dict,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    trainer = FxProcessorTrainer(
        args=args, dset=train_loader, network=network, device=device, tester=tester
    )

    trainer.training_loop()


@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
