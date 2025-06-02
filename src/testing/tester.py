from datetime import date
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

    def log_metric(self, value, name: str, step=None):
        self.wandb_run.log(
            {name: value},
            step=step if step is not None else self.it
        )

    def log_audio(self, pred, name: str, it=None):
        if it is not None:
            it = self.it 
        if self.use_wandb:
            #pred = pred.view(-1)
            #maxim = torch.max(torch.abs(pred)).detach().cpu().numpy()
            #if maxim < 1:
            #    maxim = 1
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

    def test_style(self, mode):

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
            if "pairwise" in metric:
                from evaluation.pairwise_metrics import metric_factory
                metrics_dict[metric]=metric_factory(metric, self.args.exp.sample_rate,  **self.args.tester)

        return metrics_dict
            
    def test_conditional(self, mode, exp_description=""):

        for k, test_set in self.test_set_dict.items():

            print(f"Testing on {k} set")


            assert len(test_set) != 0, "No samples found in test set"
    
            dict_y = {}
            dict_x = {}
            dict_y_hat = {}
    

            i=0
    
            for sample_y, sample_x  in tqdm(test_set):
    
    
                #print("sample_x", sample_x, "sample_y", sample_y)

                B, C, T = sample_y.shape
    

                sample_y = sample_y.to(self.device).float()
                sample_x = sample_x.to(self.device).float()

                if sample_y.dim()==2:
                    sample_y = sample_y.unsqueeze(0)
                if sample_x.dim()==2:
                    sample_x = sample_x.unsqueeze(0)
                #print("sample_y", sample_y.shape, "sample_x", sample_x.shape)
    
                preds=self.sample_conditional(mode, sample_x, B=B)

                for b in range(B):

                    if self.use_wandb:
        
                        if i < self.args.tester.wandb.num_examples_to_log:  # Log only first 10 samples
                            if self.in_training:
                                self.log_audio(preds[b], f"pred_{k}_{i}", it=self.it)  # Just log first sample
                                self.log_audio(sample_y[b], f"original_wet_{k}_{i}", it=self.it)  # Just log first sample
                                self.log_audio(sample_x[b], f"original_dry_{k}_{i}", it=self.it)  # Just log first sample
                            else:
                                self.log_audio(preds[b], f"pred_{k}", it=i)  # Just log first sample
                                self.log_audio(sample_y[b], f"original_wet_{k}", it=i)  # Just log first sample
                                self.log_audio(sample_x[b], f"original_dry_{k}", it=i)  # Just log first sample
                    
                    dict_y[i] = sample_y[b].detach().cpu().numpy()
                    dict_x[i] = sample_x[b].detach().cpu().numpy()
                    dict_y_hat[i] = preds[b].detach().cpu().numpy()

                    i += 1
            
            if self.args.tester.compute_metrics:
                for metric in self.metrics_dict.keys():
                    print(f"Computing metric {metric}")
                    result, result_dict=self.metrics_dict[metric].compute(dict_y, dict_y_hat, dict_x)
    
                    if self.use_wandb:
                        print(f"Logging metric {metric} to wandb")
                        self.log_metric(result, metric+"_"+k, step=self.it)


    def sample_conditional_style(self, mode, cond):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        #shape = [self.args.tester.unconditional.num_samples, 2,audio_len]
        shape=self.sampler.diff_params.default_shape

        cond=self.sampler.diff_params.transform_forward(cond)
        
        preds, noise_init = self.sampler.predict_conditional(shape, cond=cond, cfg_scale=self.args.tester.cfg_scale, device=self.device)

        return preds

    def sample_conditional(self, mode, cond, B=1):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        #shape = [self.args.tester.unconditional.num_samples, 2,audio_len]
        shape=self.sampler.diff_params.default_shape
        shape= [B, *shape[1:]]  # B is the batch size, we want to sample B samples

        with torch.no_grad():
            cond=self.sampler.diff_params.transform_forward(cond,is_condition=True, is_test=True)
        
        preds, noise_init = self.sampler.predict_conditional(shape, cond=cond, cfg_scale=self.args.tester.cfg_scale, device=self.device)

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
            elif m== "style_conditional_dry_vocals":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test_style(m)
            else:
                print("Warning: unknown mode: ", m)