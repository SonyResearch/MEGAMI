

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from inference.evaluator import Evaluator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import omegaconf
method_args= {
    "S1_code": "S9",
    "S2_code": "MF3wet",
    "T": 30,
    "Schurn": 10,
    "cfg_scale": 2.0,
}
method_args=omegaconf.OmegaConf.create(method_args)

inferencer=Evaluator(method="stylediffpipeline", method_args=method_args, dataset_code="TencyMastering_val_randomFx")

inferencer.run_evaluation()
