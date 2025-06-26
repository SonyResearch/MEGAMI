from datetime import date
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
from fx_model.distribution_presets.clusters_vocals import get_distributions_Cluster0, get_distributions_Cluster1

class Tester():
    def __init__(
            self, args, network, diff_params, test_set_dict=None, device=None,
            in_training=False,
    ):
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        self.device = device
        self.test_set_dict= test_set_dict

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
        
        distribution_C0_vocals = get_distributions_Cluster0(sample_rate=44100)
        distribution_C1_vocals = get_distributions_Cluster1(sample_rate=44100)

        self.effect_randomizer_C0 = {
            "vocals": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C0_vocals, device=device)
        }
        self.effect_randomizer_C1 = {
            "vocals": EffectRandomizer(sample_rate=44100, distributions_dict=distribution_C1_vocals, device=device)
        }
        self.distributions_dicts= {
            "C0": self.effect_randomizer_C0,
            "C1": self.effect_randomizer_C1
        }

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
                self.network.load_state_dict(state_dict['ema'])
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
        try:
            self.it = state_dict['it']
        except:
            self.it = 0

        print(f"loading checkpoint {self.it}")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def log_figure(self, fig, name: str, step=None):
        # Save the figure to a buffer
        #buf = io.BytesIO()
        #plt.savefig(buf, format='png')
        #buf.seek(0)

        self.wandb_run.log({name: wandb.Image(fig)}, 
                           step=step if step is not None else self.it)


    def log_metric(self, value, name: str, step=None):
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
            print("Logging audio to wandb")
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

    def sample_unconditional(self, mode):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len

        shape= self.sampler.diff_params.default_shape

        preds, noise_init = self.sampler.predict_unconditional(shape, self.device)

        if self.use_wandb:
            self.log_audio(preds[0], f"unconditional+{self.sampler.T}")  # Just log first sample
        else:
            try:
                if not self.in_training:
                    for i in range(len(preds)):
                        path_generated = utils_logging.write_audio_file(preds[i] ,
                                                                        self.args.exp.sample_rate,
                                                                        f"unconditional_{self.args.tester.wandb.pair_id}",
                                                                        path=self.paths["unconditional"])
                        path_generated_noise = utils_logging.write_audio_file(noise_init[i], self.args.exp.sample_rate,
                                                                              f"noise_{self.args.tester.wandb.pair_id}",
                                                                              path=self.paths["unconditional"])
            except:
                pass

        return preds

    def test_style_old(self, mode):

        assert len(self.test_set) != 0, "No samples found in test set"

        #print("Files will be saved in: ", self.paths[mode])

        for i, (sample_y, sample_x ) in enumerate(tqdm(self.test_set)):

            sample_y = sample_y.to(self.device).float().unsqueeze(0)
            sample_x = sample_x.to(self.device).float().unsqueeze(0)
            print("sample_y", sample_y.shape, "sample_x", sample_x.shape)

            preds=self.sample_conditional_style(mode, sample_x)

            if self.use_wandb:
                #self.log_audio(preds[0], f"pred_{i}_{self.sampler.T}")  # Just log first sample
                if i < self.args.tester.wandb.num_examples_to_log:  # Log only first 10 samples
                    self.log_audio(sample_x[0], f"original_dry+{i}")  # Just log first sample
                    self.log_audio(sample_y[0], f"original_wet+{i}")  # Just log first sample
            else:
                raise NotImplementedError
                try:
                    if not self.in_training:
                        for i in range(len(preds)):
                            path_generated = utils_logging.write_audio_file(preds[i] * sigma_data,
                                                                            self.args.exp.sample_rate,
                                                                            f"unconditional_{self.args.tester.wandb.pair_id}",
                                                                            path=self.paths["unconditional"])
                            path_generated_noise = utils_logging.write_audio_file(noise_init[i], self.args.exp.sample_rate,
                                                                                  f"noise_{self.args.tester.wandb.pair_id}",
                                                                                  path=self.paths["unconditional"])
                except:
                    pass

    def prepare_metrics(self, metrics):
        metrics_dict = {}
        for metric in metrics:
            print(f"Preparing metric {metric}")
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


    def simulate_effects(self, x, cluster):

        #first separate x in clusters

        y=x.clone() #initialize y with x

        C0= (cluster==0)

        if (C0).any():
            y[C0] = self.effect_randomizer_C0["vocals"].forward(x[C0])

        C1= (cluster==1)
        if (C1).any():
            y[C1] = self.effect_randomizer_C1["vocals"].forward(x[C1])

        return y

    def test_conditional_style(self, mode, exp_description="", input_type="dry"):

        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set")
            k+= "_" + input_type  # Add input type to the key


            assert len(test_set) != 0, "No samples found in test set"
    
            dict_y = {}
            dict_x = {}
            dict_p_hat = {}
            dict_cluster = {}
            #dict_p_target = {}
    

            i=0

            if not self.in_training:
                self.it+= 1  # Increment iteration for testing, so we can log it in wandb
    
            for sample_x, cluster  in tqdm(test_set):
    
                sample_x=sample_x.to(self.device).float()
                cluster=cluster.to(self.device)
                #print("sample_x", sample_x, "sample_y", sample_y)
                sample_y= self.simulate_effects(sample_x, cluster)

                B, C, T = sample_y.shape

                sample_y = sample_y.to(self.device).float()
                sample_x = sample_x.to(self.device).float()

                if sample_y.dim()==2:
                    sample_y = sample_y.unsqueeze(0)
                if sample_x.dim()==2:
                    sample_x = sample_x.unsqueeze(0)

                #print("sample_y", sample_y.shape, "sample_x", sample_x.shape)
                #with torch.no_grad():
                #    p_target=self.sampler.diff_params.transform_forward(sample_y,is_condition=False, is_test=True)
    
                if input_type == "dry" or input_type == "fxnorm_dry":
                    preds=self.sample_conditional_style(mode, sample_x, B=B, cluster=cluster)
                elif input_type == "fxnorm_wet":
                    preds=self.sample_conditional_style(mode, sample_y, B=B, cluster=cluster)

                for b in range(B):

                    dict_y[i] = sample_y[b].detach().cpu().numpy()
                    dict_x[i] = sample_x[b].detach().cpu().numpy()
                    dict_p_hat[i] = preds[b].detach().cpu().numpy()
                    dict_cluster[i] = cluster[b].detach().cpu()
                    #dict_p_target[i] = p_target[b].detach().cpu().numpy()

                    i += 1
            
            if self.args.tester.compute_metrics:
                for metric in self.metrics_dict.keys():
                    print(f"Computing metric {metric}")
                    result, result_dict=self.metrics_dict[metric].compute(dict_y, None, dict_x, dict_p_hat=dict_p_hat, dict_cluster=dict_cluster)
    
                    if self.use_wandb:
                        if result is not None:
                            self.log_metric(result, metric+"_"+k, step=self.it )

                        for key, value in result_dict.items():
                            if "figure" in key:
                                # log figure as an image
                                self.log_figure(value, key+"_"+k, step=self.it)
                            else:
                                self.log_metric(value, key+"_"+k, step=self.it)

                    if not self.in_training:
                        self.it+= 1  # Increment iteration for testing, so we can log it in wandb
            
    def test_conditional(self, mode, exp_description=""):

        self.it = 0
        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set")


            assert len(test_set) != 0, "No samples found in test set"
    
            dict_y = {}
            dict_x = {}
            dict_y_hat = {}
    

            i=0

            #if not self.in_training:
            #    self.it+= 1  # Increment iteration for testing, so we can log it in wandb
    
            for sample_y, sample_x  in tqdm(test_set):
    
    
                #print("sample_x", sample_x, "sample_y", sample_y)

                B, C, T = sample_y.shape
    

                sample_y = sample_y.to(self.device).float()
                sample_x = sample_x.to(self.device).float()

                if sample_y.dim()==2:
                    sample_y = sample_y.unsqueeze(0)
                if sample_x.dim()==2:
                    sample_x = sample_x.unsqueeze(0)
    
                if "baseline" in mode:
                    if mode== "baseline_dry":
                        preds=sample_x  # Just return the dry vocals as baseline
                    elif mode== "baseline_autoencoder":
                        preds=self.autoencoder_reconstruction(sample_y)  # Just return the dry vocals as baseline
                    elif mode== "baseline_random":
                        raise NotImplementedError("Baseline random sampling not implemented yet")
                        pass
                else:
                    preds=self.sample_conditional(mode, sample_x, B=B)

                    is_nan = torch.isnan(preds).any()
                    if is_nan:
                        print("NaN values found in predictions")
                        print("preds", preds.shape, "sample_y", sample_y.shape, "sample_x", sample_x.shape)
                        print("preds", preds.std(), "sample_y", sample_y.std(), "sample_x", sample_x.std())
                        #count number of NaN values in sample_x
                        num_nan = torch.sum(torch.isnan(sample_x)).item()
                        print(f"Number of NaN values in sample_x: {num_nan} of {sample_x.numel()}")


                for b in range(B):

                    if self.use_wandb:
        
                        if i < self.args.tester.wandb.num_examples_to_log:  # Log only first 10 samples
                            self.log_audio(preds[b], f"pred_{k}_{i}", it=self.it)  # Just log first sample
                            self.log_audio(sample_y[b], f"original_wet_{k}_{i}", it=self.it)  # Just log first sample
                            self.log_audio(sample_x[b], f"original_dry_{k}_{i}", it=self.it)  # Just log first sample
                    
                    dict_y[i] = sample_y[b].detach().cpu().numpy()
                    dict_x[i] = sample_x[b].detach().cpu().numpy()
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
                                self.log_metric(result, metric+"_"+k, step=self.it )
    
                            for key, value in result_dict.items():
                                print(f"Logging {key} to wandb")
                                if "figure" in key:
                                    # log figure as an image
                                    self.log_figure(value, key+"_"+k, step=self.it)
                                else:
                                    self.log_metric(value, key+"_"+k, step=self.it)
    
                        #if not self.in_training:
                        #    self.it+= 1  # Increment iteration for testing, so we can log it in wandb
                    except Exception as e:
                        print(f"Error computing metric {metric}: {e}")
                        continue
                        

    def sample_conditional_style(self, mode,  cond, B=1, cluster=None):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        #shape = [self.args.tester.unconditional.num_samples, 2,audio_len]
        shape=self.sampler.diff_params.default_shape
        shape= [B, *shape[1:]]  # B is the batch size, we want to sample B samples

        with torch.no_grad():
            cond, x_preprocessed=self.sampler.diff_params.transform_forward(cond,  is_condition=True, is_test=True, clusters=cluster)
            preds, noise_init = self.sampler.predict_conditional(shape, cond=cond, cfg_scale=self.args.tester.cfg_scale, device=self.device)

        return preds

    def autoencoder_reconstruction(self,x):


        cond_shape = x.shape

        with torch.no_grad():
            x=self.sampler.diff_params.transform_forward(x,is_condition=True, is_test=True)
            preds=self.sampler.diff_params.transform_inverse(x)

        if preds.shape[-1] != cond_shape[-1]:
            # If the shape of the predictions is not the same as the shape of the condition, we need to pad or truncate
            if preds.shape[-1] < cond_shape[-1]:
                # Pad the predictions
                preds = torch.nn.functional.pad(preds, (0, cond_shape[-1] - preds.shape[-1]))
            elif preds.shape[-1] > cond_shape[-1]:
                # Truncate the predictions
                preds = preds[..., :cond_shape[-1]]

        return preds

    def sample_conditional(self, mode, cond, B=1):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        #shape = [self.args.tester.unconditional.num_samples, 2,audio_len]
        cond_shape= cond.shape

        shape=self.sampler.diff_params.default_shape
        shape= [B, *shape[1:]]  # B is the batch size, we want to sample B samples

        with torch.no_grad():
            cond=self.sampler.diff_params.transform_forward(cond,is_condition=True, is_test=True)
        
        preds, noise_init = self.sampler.predict_conditional(shape, cond=cond, cfg_scale=self.args.tester.cfg_scale, device=self.device)

        if preds.shape[-1] != cond_shape[-1]:
            # If the shape of the predictions is not the same as the shape of the condition, we need to pad or truncate
            if preds.shape[-1] < cond_shape[-1]:
                # Pad the predictions
                preds = torch.nn.functional.pad(preds, (0, cond_shape[-1] - preds.shape[-1]))
            elif preds.shape[-1] > cond_shape[-1]:
                # Truncate the predictions
                preds = preds[..., :cond_shape[-1]]

        return preds

            


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

        if not unconditional:
            self.paths[mode + "wet_original"] = os.path.join(self.paths[mode], string + "wet_original")
            if not os.path.exists(self.paths[mode + "wet_original"]):
                os.makedirs(self.paths[mode + "wet_original"])
            self.paths[mode + "dry"] = os.path.join(self.paths[mode], string + "dry")
            if not os.path.exists(self.paths[mode + "dry"]):
                os.makedirs(self.paths[mode + "dry"])
            self.paths[mode + "emb_estimate"] = os.path.join(self.paths[mode], string + "emb_estimate")
            if not os.path.exists(self.paths[mode + "emb_estimate"]):
                os.makedirs(self.paths[mode + "emb_estimate"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"),
                  'w') as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        #print(self.args.tester.modes)
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.sample_unconditional(m)
            elif m== "conditional_dry_vocals":
                print("testing conditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional(m)
            elif m== "baseline_dry":
                print("testing baseline dry vocals")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional(m)
            elif m== "baseline_autoencoder":
                print("testing autoencoer dry vocals")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional(m)
            elif m== "style_conditional_dry_vocals":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional_style(m, input_type="dry")
            elif m== "style_conditional_fxnorm_vocals":
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_conditional_style(m, input_type="fxnorm_dry")
                self.test_conditional_style(m, input_type="fxnorm_wet")
            else:
                print("Warning: unknown mode: ", m)