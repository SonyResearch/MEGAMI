from datetime import date
import math
import io
import matplotlib.pyplot as plt
from functools import partial
import re
import torch
import os
import wandb
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import utils.log as utils_logging
import utils.training_utils as tr_utils

import soundfile as sf
import numpy as np
import torchaudio

from utils.data_utils import apply_RMS_normalization

class Tester():
    def __init__(
            self, args, network, test_set_dict=None, device=None,
            in_training=False,
    ):
        self.args = args
        self.network = network
        self.device = device
        self.test_set_dict= test_set_dict

        self.use_wandb = False  # hardcoded for now
        self.in_training = in_training

        if in_training:
            self.use_wandb = True
            # Will inherit wandb_run from Trainer
        else:  # If we use the tester in training, we will log in WandB in the Trainer() class, no need to create all these paths
            torch.backends.cudnn.benchmark = True
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.setup_wandb()


        if self.args.tester.compute_metrics:
            self.metrics_dict= self.prepare_metrics(self.args.tester.metrics)
        else:
            self.metrics_dict = {}
        

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



        self.RMS_norm=self.args.exp.RMS_norm  # Use fixed RMS for evaluation, hardcoded for now

        if self.args.exp.style_encoder_type=="FxEncoder++_DynamicFeatures":

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

                z_af,_=AFembedding.encode(x)
                #embedding is l2 normalized, normalize to unit variance
                z_af=z_af* math.sqrt(z_af.shape[-1])  # rescale to keep the same scale

                #concatenate z and z_af (rescaling with sqrt(dim) to keep the same scale)
                z_all= torch.cat([z, z_af], dim=-1)

                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z
            
            self.style_encode=fxencode_fn
        else:
            raise NotImplementedError("Only FxEncoder++_DynamicFeatures is implemented for now")

 
        if self.args.exp.apply_fxnorm:
            self.fx_normalizer= hydra.utils.instantiate(self.args.exp.fxnorm)

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run = wandb.init(project="testing" + self.args.tester.wandb.project, entity=self.args.tester.wandb.entity,
                                    config=config, tags=self.args.tester.wandb.tags)
        #wandb.watch(self.network,
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
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
            try:
                self.network.load_state_dict(state_dict['network'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        print("state_dict keys:", state_dict.keys())
        try:
            self.it = state_dict['it']
        except:
            self.it = 0
        
        print(f"loading checkpoint {self.it}")
        return tr_utils.load_state_dict(state_dict, network=self.network)

    def log_figure(self, fig, name: str, step=None):
        # Save the figure to a buffer

        self.wandb_run.log({name: wandb.Image(fig)}, 
                           step=step if step is not None else self.it)


    def log_metric(self, value, name: str, step=None):
        #print("logging metric it:", self.it, "name:", name)
        self.wandb_run.log(
            {name: value},
            step=step if step is not None else self.it
        )

    def log_audio(self, pred, name: str, it=None):
        if it is None:
            it = self.it 
        if self.use_wandb:
            pred=pred.permute(1,0)
            self.wandb_run.log(
                {name: wandb.Audio(pred.detach().cpu().numpy() , sample_rate=self.args.exp.sample_rate)},
                step=it)

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
                    from evaluation.dist_metrics_multitrack import metric_factory
                    taxonomy_ref=omegaconf.OmegaConf.create({
                        "92": "vocals",
                        "11": "drums",
                        "2": "bass"
                    })
                    metrics_dict[metric]=metric_factory(metric, self.args.exp.sample_rate,taxonomy_ref=taxonomy_ref, **self.args.tester)
            else:
                if "pairwise" in metric:
                    from evaluation.pairwise_metrics import metric_factory
                    metrics_dict[metric]=metric_factory(metric, self.args.exp.sample_rate,  **self.args.tester)
                elif "fad" in metric:
                    from evaluation.dist_metrics import metric_factory
                    metrics_dict[metric]=metric_factory(metric, self.args.exp.sample_rate, **self.args.tester)
                elif "kad" in metric:
                    from evaluation.dist_metrics import metric_factory
                    metrics_dict[metric]=metric_factory(metric, self.args.exp.sample_rate, **self.args.tester)
                elif "histogram" in metric:
                    from evaluation.feature_histograms import metric_factory
                    metrics_dict[metric] = metric_factory(metric, self.args.exp.sample_rate, **self.args.tester)

        return metrics_dict


    def test_paired(self, mode, exp_description=""):

        #self.it = 0
        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set", k)

            assert len(test_set) != 0, "No samples found in test set"
    
            dict_y = {}
            dict_x = {}
            dict_y_hat = {}
    

            i=0

            for x, y  in tqdm(test_set):
    
                B, C, T = y.shape

                x = x.to(self.device).float()
                if x.shape[-1] > self.args.exp.audio_len:
                    x = x[:, :, :self.args.exp.audio_len]
                elif x.shape[-1] < self.args.exp.audio_len:
                    raise ValueError(f"Sample length {x.shape[-1]} is less than expected {self.args.exp.audio_len}")

                if mode == "paired":
                    y = y.to(self.device).float()
                    
                    if y.shape[-1] > self.args.exp.audio_len:
                        y = y[:, :, :self.args.exp.audio_len]
                    elif y.shape[-1] < self.args.exp.audio_len:
                        raise ValueError(f"Sample length {y.shape[-1]} is less than expected {self.args.exp.audio_len}")
                
                if x.shape[1] == 2:
                    x = x.mean(dim=1, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent

                #RMS normalization of x and y
                x= apply_RMS_normalization(x, use_gate=self.args.exp.use_gated_RMSnorm)
 
                if self.args.exp.apply_fxnorm:
                    x=self.fx_normalizer(x, use_gate=self.args.exp.use_gated_RMSnorm)

                if "baseline" in mode:
                    if mode== "baseline_dry":
                        preds=x  # Just return the dry vocals as baseline
                    elif mode== "baseline_autoencoder":
                        preds=self.autoencoder_reconstruction(y)  # Just return the dry vocals as baseline
                    elif mode== "baseline_random":
                        raise NotImplementedError("Baseline random sampling not implemented yet")
                        pass
                else:
                    with torch.no_grad():
                        z=self.FXenc(y)
                        try:
                            preds=self.network(x, z)  # Get the predictions from the network
                        except Exception as e:
                            print(f"Error during inference: {e}")
                            continue
                        print("y_pred", preds.shape, preds.std(), preds.mean(), preds.min(), preds.max())

                    is_nan = torch.isnan(preds).any()
                    if is_nan:
                        num_nan = torch.sum(torch.isnan(x)).item()
                        print(f"Number of NaN values in sample_x: {num_nan} of {x.numel()}")
                    

                y= apply_RMS_normalization(y, use_gate=self.args.exp.use_gated_RMSnorm)


                for b in range(B):
                    if self.use_wandb:
                        if i < self.args.tester.wandb.num_examples_to_log:  # Log only first 10 samples
                            self.log_audio(preds[b], f"pred_wet_{k}_{mode}_{i}", it=self.it)  # Just log first sample
                            self.log_audio(y[b], f"original_wet_{k}_{mode}_{i}", it=self.it)  # Just log first sample
                            self.log_audio(x[b], f"original_dry_{k}_{mode}_{i}", it=self.it)  # Just log first sample
                    
                    dict_y[i] = y[b].detach().cpu().numpy()
                    dict_x[i] = x[b].detach().cpu().numpy()
                    dict_y_hat[i] = preds[b].detach().cpu().numpy()

                    i += 1
                
            if self.args.tester.compute_metrics:
                for metric in self.metrics_dict.keys():
                    try:
                        print(f"Computing metric {metric}")
                        result, result_dict=self.metrics_dict[metric].compute(dict_y, dict_y_hat, dict_x)
        
                        if self.use_wandb:
                            if result is not None:
                                self.log_metric(result, metric+"_"+k+"_"+mode, step=self.it )
    
                            for key, value in result_dict.items():
                                if "figure" in key:
                                    # log figure as an image
                                    self.log_figure(value, key+"_"+k+"_"+mode, step=self.it)
                                else:
                                    self.log_metric(value, key+"_"+k+"_"+mode, step=self.it)
    
                    except Exception as e:
                        print(f"Error computing metric {metric}: {e}")
                        continue
                        
    def prepare_directories(self, mode, unconditional=False, string=None):

        today = date.today()
        self.paths = {}
        if "overriden_name" in self.args.tester.keys() and self.args.tester.overriden_name is not None:
            self.path_sampling = os.path.join(self.args.model_dir, self.args.tester.overriden_name)
        else:
            self.path_sampling = os.path.join(self.args.model_dir, 'test' + today.strftime("%d_%m_%Y"))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.paths[mode] = os.path.join(self.path_sampling, mode, self.args.exp.exp_name)

        if not os.path.exists(self.paths[mode]):
            os.makedirs(self.paths[mode])
        if string is None:
            string = ""

        self.paths[mode + "wet_original"] = os.path.join(self.paths[mode], string + "wet_original")
        if not os.path.exists(self.paths[mode + "wet_original"]):
            os.makedirs(self.paths[mode + "wet_original"])
        self.paths[mode + "dry"] = os.path.join(self.paths[mode], string + "dry")
        if not os.path.exists(self.paths[mode + "dry"]):
            os.makedirs(self.paths[mode + "dry"])
        self.paths[mode + "wet_estimate"] = os.path.join(self.paths[mode], string + "wet_estimate")
        if not os.path.exists(self.paths[mode + "wet_estimate"]):
            os.makedirs(self.paths[mode + "wet_estimate"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"),
                  'w') as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        for m in self.args.tester.modes:
            if m == "paired":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.test_paired(m)
            else:
                print("Warning: unknown mode: ", m)
