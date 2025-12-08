# Copyright (c) 2025 Sony Research
# Licensed under CC BY-NC-SA 4.0
# See LICENSE file for details

import os
import sys
import re
import json
import hydra

import time
import copy
import numpy as np
import torch
from glob import glob
import wandb
import omegaconf

from utils.torch_utils import training_stats
import utils.log as utils_logging
import utils.training_utils as t_utils


from utils.collators import collate_multitrack, collate_multitrack_paired

from inference.validator_FxGenerator import ValidatorFxGenerator
import copy

# ----------------------------------------------------------------------------


class FxGeneratorTrainer:
    def __init__(
        self,
        args=None,
        dset=None,
        network=None,
        diff_params=None,
        tester=None,
        device="cpu",
    ):
        """
        Constructor for the FxGeneratorTrainer class.
        Args:
            args: omegaconf dictionary containing the experiment configuration.
            dset: training dataset dataloader.
            network: the neural network to be trained.
            diff_params: the diffusion parameters.
            tester: the tester object for validation during training.
            device: the device to be used for training (cpu or cuda).
        """

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

        print("total_params: ", self.total_params / 1e6, "M")

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

        self.network = self.network.to(self.device)

        # Logger Setup
        if self.args.logging.log:
            self.setup_wandb()
            if self.tester is not None:
                self.tester.setup_wandb_run(self.wandb_run)
            self.setup_logging_variables()

        self.skip_val = False  # This is used to skip validation during training, useful for debugging

        try:
            if self.args.exp.skip_first_val:
                self.skip_val = True
        except Exception as e:
            print(e)
            pass

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

    def setup_logging_variables(self):
        """
        Set up the variables used for logging the losses by sigma bins.
        """
        if self.diff_params.type == "FM":
            self.sigma_bins = np.linspace(
                self.args.diff_params.sde_hp.t_min,
                self.args.diff_params.sde_hp.t_max,
                num=self.args.logging.num_sigma_bins,
            )
        elif self.diff_params.type == "ve_karras":
            self.sigma_bins = np.logspace(
                np.log10(self.args.diff_params.sde_hp.sigma_min),
                np.log10(self.args.diff_params.sde_hp.sigma_max),
                num=self.args.logging.num_sigma_bins,
                base=10,
            )

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the network, ema and optimizer.
        """
        return t_utils.load_state_dict(
            state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer
        )

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        """
        Loads the latest checkpoint from the model directory if available.
        """
        # Resume training from latest checkpoint available in the output director
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
        Prepare the state dictionary for saving.
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

    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        torch.nan_to_num(error)
        error_mean = error.mean()
        error_mean = error_mean.detach().cpu().numpy()

        training_stats.report("loss", error_mean)

        for i in range(len(self.sigma_bins)):
            if i == 0:
                mask = sigma <= self.sigma_bins[i]
            elif i == len(self.sigma_bins) - 1:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i - 1])

            else:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i - 1])
            mask = mask.squeeze(-1).cpu()
            if mask.sum() > 0:
                idx = torch.where(mask)[0]

                error_sigma_i = error[idx].mean()
                training_stats.report(
                    "error_sigma_" + str(self.sigma_bins[i]),
                    error_sigma_i.detach().cpu().numpy(),
                )

    def get_batch(self):
        """Get an audio example from dset and apply the transform (spectrogram + compression)"""
        data = next(self.dset)

        assert (
            len(data) == self.args.exp.batch_size
        ), "Batch size mismatch, expected {}, got {}".format(
            self.args.exp.batch_size, len(data)
        )

        if len(data[0]) == 4:
            collated_data = collate_multitrack_paired(
                data,
                max_tracks=self.args.exp.max_tracks,
                sample_rate=self.args.exp.sample_rate,
                segment_length=self.args.exp.audio_len,
                device=self.device,
            )  # collate the data, this will pad the data to the maximum number of tracks in the batch
        elif len(data[0]) == 3:
            collated_data = collate_multitrack(
                data,
                max_tracks=self.args.exp.max_tracks,
                sample_rate=self.args.exp.sample_rate,
                segment_length=self.args.exp.audio_len,
                device=self.device,
            )  # collate the data, this will pad the data to the maximum number of tracks in the batch
        else:
            raise ValueError(
                "Data must have 3 or 4 elements, got {}".format(len(data[0]))
            )

        y = collated_data[
            "y"
        ]  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
        masks = collated_data[
            "masks"
        ]  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

        y = y.to(self.device)  # move the data to the device
        masks = masks.to(self.device)

        return y, masks

    def train_step(self):
        """Training step"""

        self.optimizer.zero_grad()

        a = time.time()

        y, masks = self.get_batch()

        if torch.isnan(y).any():
            raise ValueError("Input tensor contains NaN values")

        error, sigma, x_norm, y = self.diff_params.loss_fn(
            self.network, sample=y, ema=self.ema, clusters=None, masks=masks
        )

        loss = error.mean(dim=1)
        loss = loss.mean()

        if not torch.isnan(loss).any():
            loss.backward()

            if self.args.exp.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.args.exp.max_grad_norm
                )

            # Update weights.
            self.optimizer.step()

            print("iteration", self.it, "loss", loss.item(), "time", time.time() - a)

            if self.args.logging.log:
                self.process_loss_for_logging(error, sigma)

        else:
            print("loss is NaN, skipping step")

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

        training_stats.default_collector.update()
        loss_mean = training_stats.default_collector.mean("loss")
        self.wandb_run.log({"loss": loss_mean}, step=self.it)

        sigma_means = []
        sigma_stds = []
        for i in range(len(self.sigma_bins)):
            a = training_stats.default_collector.mean(
                "error_sigma_" + str(self.sigma_bins[i])
            )
            # replace NaN with 0
            if np.isnan(a):
                a = 0
            a = np.float32(a)  # convert to float32
            sigma_means.append(a)

            self.wandb_run.log(
                {"error_sigma_" + str(self.sigma_bins[i]): a}, step=self.it
            )
            a = training_stats.default_collector.std(
                "error_sigma_" + str(self.sigma_bins[i])
            )
            if np.isnan(a):
                a = 0
            a = np.float32(a)  # convert to float32
            sigma_stds.append(a)

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
        """
        Log an audio example to wandb.
        """
        string = name + "_" + self.args.tester.name

        # dividing by 2 to avoid clipping
        x = x / 2.0

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
            step=self.it,
        )

    def training_loop(self):
        """
        Main training loop.
        """

        self.logged_audio_examples = 0

        while True:
            try:
                self.train_step()
            except Exception as e:
                print("Error during training step:", e)
                print("Skipping this step")
                #    # If an error occurs, we skip the step and continue training
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
                    # pass
                    self.heavy_logging()

            if (
                self.it > 0
                and self.it % self.args.logging.log_interval == 0
                and self.args.logging.log
            ):
                self.easy_logging()

            # Update state.
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
    np.random.seed(seed)


def _main(args):

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
        prefetch_factor=20,
        collate_fn=lambda x: x,
    )
    train_loader = iter(train_loader)

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

    diff_params = hydra.utils.instantiate(
        args.diff_params
    )  # instantiate in trainer better


    print("args.dset.validation", args.dset.validation)
    if args.dset.validation is not None:

        raise NotImplementedError("Validation not implemented for FxGeneratorTrainer. Set validation to None.")

        val_set_dict = {}
        val_set = hydra.utils.instantiate(args.dset.validation)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=args.exp.val_batch_size, collate_fn=lambda x: x
        )
        val_set_dict[args.dset.validation.mode] = val_loader
    
        # try:
        val_set_2 = hydra.utils.instantiate(args.dset.validation_2)
        val_loader_2 = torch.utils.data.DataLoader(
            dataset=val_set_2, batch_size=args.exp.val_batch_size, collate_fn=lambda x: x
        )
        val_set_dict[args.dset.validation_2.mode] = val_loader_2
    
        # Tester
        args.tester.sampling_params.same_as_training = True  # Make sure that we use the same HP for sampling as the ones used in training
        args.tester.wandb.use = False  # Will do that in training

        network_tester = copy.deepcopy(network).eval().requires_grad_(False)

        tester = ValidatorFxGenerator(
            args,
            network_tester,
            diff_params,
            device=device,
            in_training=True,
            test_set_dict=val_set_dict,
        )
    else:
        tester=None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    trainer = FxGeneratorTrainer(
        args=args,
        dset=train_loader,
        network=network,
        diff_params=diff_params,
        tester=tester,
        device=device,
    )  # This works

    trainer.training_loop()


@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
