import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from evaluation.evaluator import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import omegaconf
method_args= {
    "S1_code": "S9",
    "S2_code": "MF3wet",
    "T": 30,
    "Schurn": 5,
    "cfg_scale": 1.0,
}

method_args=omegaconf.OmegaConf.create(method_args)

evaluator=Evaluator(method="stylediffpipeline", method_args=method_args, dataset_code="MDX_TM_benchmark", extra_id="27_07", path_results="/data2/eloi/results")

evaluator.run_evaluation_paired()
