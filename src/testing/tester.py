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
import utils.testing_utils as tt_utils

import soundfile as sf
import numpy as np
import torchaudio


class Tester():
    def __init__(
            self, args, network, diff_params, inference_train_set=None, inference_test_set=None, device=None,
            in_training=False,
            training_wandb_run=None
    ):
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        self.device = device
        self.inference_train_set = inference_train_set
        self.inference_test_set = inference_test_set
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

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run = wandb.init(project="testing" + self.args.exp.wandb.project, entity=self.args.exp.wandb.entity,
                                    config=config)
        wandb.watch(self.network,
                    log_freq=self.args.logging.heavy_log_interval)

        self.wandb_run.name = self.args.exp.exp_name 
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

    def log_audio(self, pred, name: str):
        if self.use_wandb:
            print(pred.shape)
            pred = pred.view(-1)
            maxim = torch.max(torch.abs(pred)).detach().cpu().numpy()
            if maxim < 1:
                maxim = 1
            self.wandb_run.log(
                {name: wandb.Audio(pred.detach().cpu().numpy() / maxim, sample_rate=self.args.exp.sample_rate)},
                step=self.it)

            if self.args.logging.log_spectrograms:
                raise NotImplementedError

    # ------------- UNCONDITIONAL SAMPLING ---------------#

    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################

    def sample_unconditional(self, mode):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        shape = [self.args.tester.unconditional.num_samples, audio_len]
        preds, noise_init = self.sampler.predict_unconditional(shape, self.device)
        sigma_data = self.sampler.diff_params.sigma_data

        if self.use_wandb:
            # preds=preds/torch.max(torch.abs(preds))
            self.log_audio(preds[0], f"unconditional+{self.sampler.T}")  # Just log first sample
            # self.log_unconditional_metrics(preds) #But compute metrics on several
        else:
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

        return preds

    # ------------- CONDITIONAL SAMPLING ---------------#

    def run_inference(self, x_ref, params_dict, mode, blind=False, oracle=False, i=0):

            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.prepare_directories(mode, unconditional=False, blind=blind, string=str(i))

            path_dict = {
                "original": self.paths[mode + "original"],
                "degraded": self.paths[mode + "degraded"],
                "reconstructed": self.paths[mode + "reconstructed"],
                "operator_ref": self.paths[mode + "operator_ref"]
            }
            if blind:
                path_dict["operator"] = self.paths[mode + "operator"]

            # if original file exists, we skip
            if os.path.exists(
                    os.path.join(self.paths[mode + "original"], "x.wav")) and not self.args.tester.set.overwrite:
                print("skipping", i)
                return

            if isinstance(x_ref, list):
                if self.args.tester.operator_ref is None:
                    operator_ref=None
                    y_ref=[]
                    for i in range(len(x_ref)):
                        y_ref.append(params_dict[i]["y"])
                else:
                    operator_ref=hydra.utils.instantiate(self.args.tester.operator_ref, params_dict=params_dict[0]) #assuming all the same
                    y_ref=[]
                    for i in range(len(x_ref)):
                        y_ref.append(operator_ref.forward(x_ref[i]))
                
                x_ref=torch.stack(x_ref)
                y_ref=torch.stack(y_ref)
                x_ref=x_ref.view(x_ref.shape[0], -1).to(device)
                y_ref=y_ref.view(y_ref.shape[0], -1).to(device)

            else:
                if self.args.tester.operator_ref is None:
                    # account for PairedDataset, no operator_ref available
                    operator_ref=None
                    y_ref=params_dict["y"]
                else:
                    operator_ref=hydra.utils.instantiate(self.args.tester.operator_ref, params_dict=params_dict)
                    y_ref=operator_ref.forward(x_ref)

                x_ref=x_ref.view(1, -1).to(device)
                y_ref=y_ref.view(1, -1).to(device)

            if self.args.tester.preprocessing is not None:
                preprocesser=hydra.utils.instantiate(self.args.tester.preprocessing)
                y_ref=preprocesser(y_ref)

            #TODO: Maybe I want to adapt the operator parameters to something dependent of x_ref (e.g. SDR)


            if blind:
                #from operators.nablaFX_operator import NablaFXOperator as Nabk
                #operator_blind=Nabk()
                operator_blind = hydra.utils.instantiate(self.args.tester.operator_blind)


            logging_callback = hydra.utils.instantiate(self.args.tester.logging_callback)
            evaluation_callback = hydra.utils.instantiate(self.args.tester.evaluation_callback)
            init_callback = hydra.utils.instantiate(self.args.tester.init_callback)
    
            my_init_callback = partial(init_callback, args=self.args)
            my_logging_callback = logging_callback
    

            my_evaluation_callback = partial(evaluation_callback, x_test=None, y_test=None, args=self.args,
                                         paths=path_dict, blind=blind)


            pred = self.sampler.predict_conditional(y_ref, operator_blind=operator_blind if blind else operator_ref,
                                                reference=x_ref, blind=blind, logging_callback=my_logging_callback,
                                                init_callback=my_init_callback,
                                                operator_ref=operator_ref,
                                                evaluation_callback=my_evaluation_callback,
                                                save_path=self.paths[mode],
                                                oracle=oracle)
            return pred

    def test(self, mode, blind=False, single_example=True, oracle=False):

        assert len(self.inference_train_set) != 0, "No samples found in test set"

        print("Files will be saved in: ", self.paths[mode])

        if single_example:

            for i, (x_ref, params_dict) in enumerate(tqdm(self.inference_train_set)):
    
                self.run_inference(x_ref, params_dict, mode, blind=blind, oracle=oracle, i=i)
        else:

            x=[]
            params_dict_list=[]
            for i, (x_ref, params_dict) in enumerate(tqdm(self.inference_train_set)):
                x.append(x_ref)
                params_dict_list.append(params_dict)
            
            self.run_inference(x, params_dict_list, mode, blind=blind, oracle=oracle, i=0)


            


    def prepare_directories(self, mode, unconditional=False, blind=False, string=None):

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
            self.paths[mode + "original"] = os.path.join(self.paths[mode], string + "original")
            if not os.path.exists(self.paths[mode + "original"]):
                os.makedirs(self.paths[mode + "original"])
            self.paths[mode + "degraded"] = os.path.join(self.paths[mode], string + "degraded")
            if not os.path.exists(self.paths[mode + "degraded"]):
                os.makedirs(self.paths[mode + "degraded"])
            self.paths[mode + "reconstructed"] = os.path.join(self.paths[mode], string + "reconstructed")
            if not os.path.exists(self.paths[mode + "reconstructed"]):
                os.makedirs(self.paths[mode + "reconstructed"])
            self.paths[mode + "operator_ref"] = os.path.join(self.paths[mode], string + "operator_ref")
            if not os.path.exists(self.paths[mode + "operator_ref"]):
                os.makedirs(self.paths[mode + "operator_ref"])
            if blind:
                self.paths[mode + "operator"] = os.path.join(self.paths[mode], string + "operator")
                if not os.path.exists(self.paths[mode + "operator"]):
                    os.makedirs(self.paths[mode + "operator"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"),
                  'w') as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        print(self.args.tester.modes)
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.sample_unconditional(m)
            elif m == "blind":
                assert self.inference_train_set is not None
                print("testing blind distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=True)
                    self.save_experiment_args(m)

                self.test(m, blind=True, single_example=True)
            elif m == "blind_set":
                assert self.inference_train_set is not None
                print("testing blind distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=True)
                    self.save_experiment_args(m)

                self.test(m, blind=True, single_example=False)
            elif m == "informed_set":
                assert self.inference_train_set is not None
                print("testing blind distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=False)
                    self.save_experiment_args(m)

                self.test(m, blind=False, single_example=False)
            elif m == "informed":
                assert self.inference_train_set is not None
                print("testing informed distortion")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False)
                    self.save_experiment_args(m)
                self.test(m, blind=False, single_example=True)
            elif m == "oracle":
                assert self.inference_train_set is not None
                print("testing oracle distortion ")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=False, blind=True)
                    self.save_experiment_args(m)

                self.test(m, blind=True, single_example=True, oracle=True)
            else:
                print("Warning: unknown mode: ", m)