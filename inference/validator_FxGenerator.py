from datetime import date
import io
import re
import torch
import os
import wandb
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import utils.training_utils as tr_utils

from utils.collators import collate_multitrack_paired


class ValidatorFxGenerator:
    def __init__(
        self,
        args,
        network,
        diff_params,
        test_set_dict=None,
        device=None,
        in_training=False,
    ):
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        self.device = device
        self.test_set_dict = test_set_dict

        self.use_wandb = False  # hardcoded for now
        self.in_training = in_training
        self.sampler = hydra.utils.instantiate(
            args.tester.sampler, self.network, self.diff_params, self.args
        )

        if in_training:
            self.use_wandb = True
            # Will inherit wandb_run from Trainer
        else:  # If we use the tester in training, we will log in WandB in the Trainer() class, no need to create all these paths
            torch.backends.cudnn.benchmark = True
            if self.device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

            self.setup_wandb()

        self.taxonomy_ref = {
            "9120": "FemaleLeadVocals",
            "1110": "AcousticDrums",
            "2200": "ElectricBass",
        }

        if self.args.tester.compute_metrics:
            self.metrics_dict = self.prepare_metrics(self.args.tester.metrics)
        else:
            self.metrics_dict = {}

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run = wandb.init(
            project="testing" + self.args.tester.wandb.project,
            entity=self.args.tester.wandb.entity,
            config=config,
            tags=self.args.tester.wandb.tags,
        )
        # wandb.watch(self.network,
        #            log_freq=self.args.logging.heavy_log_interval)

        self.wandb_run.name = self.args.tester.wandb.run_name
        self.use_wandb = True

    def setup_wandb_run(self, run):
        # get the wandb run object from outside (in trainer.py or somewhere else)
        self.wandb_run = run
        self.use_wandb = True

    def load_latest_checkpoint(self):
        # load the latest checkpoint from self.args.model_dir
        try:
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

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt",
                map_location=self.device,
            )
            try:
                self.network.load_state_dict(state_dict["ema"])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict["model"], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.it = state_dict["it"]
        except:
            self.it = 0

        print(f"loading checkpoint {self.it}")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def log_figure(self, fig, name: str, step=None):
        self.wandb_run.log(
            {name: wandb.Image(fig)}, step=step if step is not None else self.it
        )

    def log_metric(self, value, name: str, step=None):
        self.wandb_run.log({name: value}, step=step if step is not None else self.it)

    def log_audio(self, pred, name: str, it=None):
        if it is None:
            it = self.it
        if self.use_wandb:
            print("Logging audio to wandb")
            pred = pred.permute(1, 0)
            self.wandb_run.log(
                {
                    name: wandb.Audio(
                        pred.detach().cpu().numpy(),
                        sample_rate=self.args.exp.sample_rate,
                    )
                },
                step=it,
            )

            if self.args.logging.log_spectrograms:
                raise NotImplementedError

    # ------------- UNCONDITIONAL SAMPLING ---------------#

    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################

    def prepare_metrics(self, metrics):
        metrics_dict = {}
        for metric in metrics:
            print(f"Preparing metric {metric}")
            if "multitrack" in metric:
                if "kad" in metric:
                    from utils.evaluation.dist_metrics_multitrack import metric_factory

                    taxonomy_ref = omegaconf.OmegaConf.create(self.taxonomy_ref)
                    metrics_dict[metric] = metric_factory(
                        metric,
                        self.args.exp.sample_rate,
                        taxonomy_ref=taxonomy_ref,
                        **self.args.tester,
                    )

        return metrics_dict

    def test_conditional_style_multitrack(
        self, mode, exp_description="", input_type="dry"
    ):

        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set")
            k += "_" + input_type  # Add input type to the key

            assert len(test_set) != 0, "No samples found in test set"

            dict_y = {}
            dict_p_hat = {}
            dict_taxonomy = {}

            i = 0

            if not self.in_training:
                self.it += (
                    1  # Increment iteration for testing, so we can log it in wandb
                )

            for data in tqdm(test_set):

                collated_data = collate_multitrack_paired(data)

                y = collated_data["y"].to(
                    self.device
                )  # y is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                taxonomy = collated_data[
                    "taxonomies"
                ]  # taxonomy is a list of lists of taxonomies, each list is a track, each taxonomy is a string of 2 digits
                masks = collated_data["masks"].to(
                    self.device
                )  # masks is a tensor of shape [B, N] where B is the batch size and N is the number of tracks, it is used to mask the tracks that are not present in the batch

                B, N, C, T = y.shape
                y = y.to(self.device).float()

                if input_type != "wet":
                    x = collated_data["x"].to(
                        self.device
                    )  # x is a tensor of shape [B, N, C, L] where B is the batch size, N is the number of tracks, C is the number of channels and L is the length of the audio
                    assert (
                        y.shape == x.shape
                    ), "sample_y and sample_x must have the same shape"
                    x = x.to(self.device).float()

                if input_type == "dry" or input_type == "fxnorm_dry":
                    preds = self.sample_conditional_style_multitrack(
                        mode,
                        x,
                        B=B,
                        N=N,
                        masks=masks,
                        taxonomy=taxonomy,
                        input_type=input_type,
                    )
                elif input_type == "fxnorm_wet" or input_type == "wet":
                    preds = self.sample_conditional_style_multitrack(
                        mode,
                        y,
                        B=B,
                        N=N,
                        masks=masks,
                        taxonomy=taxonomy,
                        input_type=input_type,
                    )

                for b in range(B):
                    indexes = masks[b].nonzero(as_tuple=True)[
                        0
                    ]  # Get the indexes of the tracks that are present in the batch

                    dict_y[i] = (
                        y[b, indexes].detach().cpu().numpy()
                    )  # Store the wet audio for each track
                    # dict_x[i] = sample_x[b].detach().cpu().numpy()
                    dict_p_hat[i] = preds[b, indexes].detach().cpu().numpy()
                    taxonomies = taxonomy[b]  # Get the taxonomy for the current batch
                    dict_taxonomy[i] = [
                        taxonomies[k] for k in indexes
                    ]  # Store the taxonomy for each track that is present in the batch

                    i += 1

            if self.args.tester.compute_metrics:
                for metric in self.metrics_dict.keys():
                    print(f"Computing metric {metric}")
                    result, result_dict = self.metrics_dict[metric].compute(
                        dict_y,
                        None,
                        None,
                        dict_p_hat=dict_p_hat,
                        dict_cluster=None,
                        dict_taxonomy=dict_taxonomy,
                    )

                    if self.use_wandb:
                        if result is not None:
                            self.log_metric(result, metric + "_" + k, step=self.it)

                        for key, value in result_dict.items():
                            if "figure" in key:
                                # log figure as an image
                                self.log_figure(value, key + "_" + k, step=self.it)
                            else:
                                self.log_metric(value, key + "_" + k, step=self.it)

                    if not self.in_training:
                        self.it += 1  # Increment iteration for testing, so we can log it in wandb

    def sample_conditional_style_multitrack(
        self,
        mode,
        cond,
        B=1,
        N=3,
        cluster=None,
        taxonomy=None,
        masks=None,
        input_type="dry",
    ):

        audio_len = (
            self.args.exp.audio_len
            if not "audio_len" in self.args.tester.unconditional.keys()
            else self.args.tester.unconditional.audio_len
        )
        shape = self.sampler.diff_params.default_shape
        shape = [B, N, *shape[2:]]  # B is the batch size, we want to sample B samples

        with torch.no_grad():
            is_wet = "wet" in input_type

            cond, x_preprocessed = self.sampler.diff_params.transform_forward(
                cond,
                is_condition=True,
                is_test=True,
                clusters=cluster,
                taxonomy=taxonomy,
                masks=masks,
                is_wet=is_wet,
            )

            preds, noise_init = self.sampler.predict_conditional(
                shape,
                cond=cond,
                cfg_scale=self.args.tester.cfg_scale,
                device=self.device,
                taxonomy=taxonomy,
                masks=masks,
            )

        return preds

    def prepare_directories(self, mode, unconditional=False, string=None):

        today = date.today()
        self.paths = {}
        if (
            "overriden_name" in self.args.tester.keys()
            and self.args.tester.overriden_name is not None
        ):
            self.path_sampling = os.path.join(
                self.args.model_dir, self.args.tester.overriden_name
            )
        else:
            self.path_sampling = os.path.join(
                self.args.model_dir, "test" + today.strftime("%d_%m_%Y")
            )
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.paths[mode] = os.path.join(
            self.path_sampling, mode, self.args.exp.exp_name
        )

        if not os.path.exists(self.paths[mode]):
            os.makedirs(self.paths[mode])
        if string is None:
            string = ""

        if not unconditional:
            self.paths[mode + "wet_original"] = os.path.join(
                self.paths[mode], string + "wet_original"
            )
            if not os.path.exists(self.paths[mode + "wet_original"]):
                os.makedirs(self.paths[mode + "wet_original"])
            self.paths[mode + "dry"] = os.path.join(self.paths[mode], string + "dry")
            if not os.path.exists(self.paths[mode + "dry"]):
                os.makedirs(self.paths[mode + "dry"])
            self.paths[mode + "emb_estimate"] = os.path.join(
                self.paths[mode], string + "emb_estimate"
            )
            if not os.path.exists(self.paths[mode + "emb_estimate"]):
                os.makedirs(self.paths[mode + "emb_estimate"])

    def save_experiment_args(self, mode):
        with open(
            os.path.join(self.paths[mode], ".argv"), "w"
        ) as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        for m in self.args.tester.modes:

            if m == "style_conditional_dry_multitrack":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional_style_multitrack(m, input_type="dry")
            elif m == "style_conditional_fxnorm_multitrack":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional_style_multitrack(m, input_type="fxnorm_wet")
                self.test_conditional_style_multitrack(m, input_type="fxnorm_dry")
            elif m == "style_conditional_wet_multitrack":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional_style_multitrack(m, input_type="wet")
            else:
                print("Warning: unknown mode: ", m)
