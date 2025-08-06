import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from evaluation.evaluator import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import omegaconf
method_args= {
    "S1_code": "S9v6",
    "S2_code": "MF3wetv6",
    "T": 50,
    "Schurn": 5,
    "cfg_scale": 1.0,
}

method_args=omegaconf.OmegaConf.create(method_args)

evaluator=Evaluator(method="stylediffpipeline", method_args=method_args, dataset_code="MDX_TM_benchmark", extra_id="cfg_1_T50_churn5", path_results="/data2/eloi/results")

evaluator.run_evaluation_paired()
