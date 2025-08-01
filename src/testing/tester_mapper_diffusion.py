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

from fx_model.fx_pipeline import EffectRandomizer
#from fx_model.distribution_presets.clusters_vocals import get_distributions_Cluster0, get_distributions_Cluster1
#from fx_model.apply_effects_multitrack_utils import simulate_effects

from fx_model.distribution_presets.clusters_multitrack import get_distributions_Cluster0_vocals, get_distributions_Cluster1_vocals, get_distributions_Cluster0_bass, get_distributions_Cluster1_bass, get_distributions_Cluster0_drums, get_distributions_Cluster1_drums

#from fx_model.distribution_presets.uniform_RMSnorm import get_distributions_uniform
from fx_model.distribution_presets.uniform import get_distributions_uniform

from utils.collators import collate_multitrack_sim

class Tester():
    def __init__(
            self, args, network, test_set_dict=None, device=None,
            in_training=False,
            diff_params=None,
    ):
        self.args = args
        self.network = network
        self.device = device
        self.test_set_dict= test_set_dict
        self.diff_params = copy.copy(diff_params)

        self.use_wandb = False  # hardcoded for now
        self.in_training = in_training
        self.sampler = hydra.utils.instantiate(args.tester.sampler, self.network, self.diff_params, self.args)

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


                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize
                return torch.nn.functional.normalize(z_all, dim=-1, p=2)
            
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
                return torch.nn.functional.normalize(z_all, dim=-1, p=2)

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

                return z_all/math.sqrt(z_all.shape[-1])  # L2 normalize by dividing by sqrt(dim) to keep the same scale

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

                #z=z* math.sqrt(z.shape[-1]) 
                #z_af=z_af* math.sqrt(z_af.shape[-1])



                z_all= torch.cat([z, z_af], dim=-1)

                #now L2 normalize

                #return torch.nn.functional.normalize(z_all, dim=-1, p=2)
                norm_z= z_all/ math.sqrt(z_all.shape[-1])  # normalize by dividing by sqrt(dim) to keep the same scale

                return norm_z

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
        #buf = io.BytesIO()
        #plt.savefig(buf, format='png')
        #buf.seek(0)
        #print("logging figure it:", self.it, "name:", name)

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
            #pred = pred.view(-1)
            #maxim = torch.max(torch.abs(pred)).detach().cpu().numpy()
            #if maxim < 1:
            #    maxim = 1
            #print("Logging audio to wandb")
            pred=pred.permute(1,0)
            #print("logging audio it:", it, "name:", name, "pred shape:", pred.shape)
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

    def apply_RMS_normalization(self, x):

        RMS= torch.tensor(self.RMS_norm, device=x.device).view(1, 1, 1).repeat(x.shape[0],1,1)  # Use fixed RMS for evaluation

        x_RMS=20*torch.log10(torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True).mean(dim=-2, keepdim=True)))

        gain= RMS - x_RMS
        gain_linear = 10 ** (gain / 20 + 1e-6)  # Convert dB gain to linear scale, adding a small value to avoid division by zero
        x=x* gain_linear.view(-1, 1, 1)

        return x

    def apply_random_effects(self, x):

        y=self.effect_randomizer_uniform.forward(x)

        return y
            
    def sample_conditional(self, mode, x, z, B=1):

        cond_shape= x.shape

        shape=self.sampler.diff_params.default_shape
        shape= [B, *shape[1:]]  # B is the batch size, we want to sample B samples

        preds, noise_init = self.sampler.predict_conditional(shape, cond=x, embedding=z,cfg_scale=self.args.tester.cfg_scale, device=self.device)

        if preds.shape[-1] != cond_shape[-1]:
            # If the shape of the predictions is not the same as the shape of the condition, we need to pad or truncate
            if preds.shape[-1] < cond_shape[-1]:
                # Pad the predictions
                preds = torch.nn.functional.pad(preds, (0, cond_shape[-1] - preds.shape[-1]))
            elif preds.shape[-1] > cond_shape[-1]:
                # Truncate the predictions
                preds = preds[..., :cond_shape[-1]]

        return preds

    def test_paired(self, mode, exp_description=""):

        #self.it = 0
        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set")


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
                elif mode == "random_effects":
                    try:
                        y = self.apply_random_effects(x)
                    except Exception as e:
                        print(f"Error applying random effects: {e}")
                        continue
                
                if x.shape[1] == 2:
                    x = x.mean(dim=1, keepdim=True)  # expand to [B*N, 1, L] to keep the shape consistent

                #RMS normalization of x and y
                x= self.apply_RMS_normalization(x)  # apply RMS normalization to x


    
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
                        #print("y", y.shape, y.std(), y.mean(), y.min(), y.max())
                        #print("x",x.shape, x.std(), x.mean(), x.min(), x.max())
                        z=self.FXenc(y)
                        #print("z", z.shape, z.std(), z.mean(), z.min(), z.max())
                        #try:
                        preds=self.sample_conditional(mode, x, z, B=B)
                        #except Exception as e:
                        #    print(f"Error during inference: {e}")
                        #    continue
                        print("y_pred", preds.shape, preds.std(), preds.mean(), preds.min(), preds.max())

                    is_nan = torch.isnan(preds).any()
                    if is_nan:
                        print("NaN values found in predictions")
                        #print("preds", preds.shape, "sample_y", y.shape, "sample_x", x.shape)
                        #print("preds", preds.std(), "sample_y", y.std(), "sample_x", x.std())
                        #count number of NaN values in sample_x
                        num_nan = torch.sum(torch.isnan(x)).item()
                        print(f"Number of NaN values in sample_x: {num_nan} of {x.numel()}")
                    

                if self.args.exp.rms_normalize_y:
                    rms_y= torch.sqrt(torch.mean(y**2, dim=(-1), keepdim=True))
                    # normalize preds to the same RMS as y
                    preds= preds * (rms_y / torch.sqrt(torch.mean(preds**2, dim=(-1), keepdim=True) + 1e-6))


                for b in range(B):
                    if self.use_wandb:
        
                        if i < self.args.tester.wandb.num_examples_to_log:  # Log only first 10 samples
                            print(preds[b].shape, y[b].shape, x[b].shape)
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
        
                        print("using wandb:", self.use_wandb)
                        if self.use_wandb:
                            if result is not None:
                                print(f"Logging metric {metric} to wandb")
                                self.log_metric(result, metric+"_"+k+"_"+mode, step=self.it )
    
                            for key, value in result_dict.items():
                                print(f"Logging {key} to wandb")
                                if "figure" in key:
                                    # log figure as an image
                                    self.log_figure(value, key+"_"+k+"_"+mode, step=self.it)
                                else:
                                    self.log_metric(value, key+"_"+k+"_"+mode, step=self.it)
    
                        #if not self.in_training:
                        #    self.it+= 1  # Increment iteration for testing, so we can log it in wandb
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
        #print(self.args.tester.modes)
        for m in self.args.tester.modes:

            if m == "paired":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.test_paired(m)
            elif m== "random_effects":
                print("testing conditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_paired(m)
            else:
                print("Warning: unknown mode: ", m)
