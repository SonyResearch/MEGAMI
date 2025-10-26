#!/bin/bash

# Example usage:
# bash run_eval.sh FxGenerator_code=public FxProcessor_code=public T=30 Schurn=0 cfg_scale=1.0 dataset=MDX_TM_benchmark extra_id=test_name

# Default values (can be overridden by passing arguments)
#FxGenerator_code="public"
#FxProcessor_code="public"

FxGenerator_code="internal_TencyDB"
FxProcessor_code="internal_TencyMastering"

T=30
Schurn=0
cfg_scale=1.0
dataset="MDX_TM_benchmark"
extra_id="default_run"
path_results="/scratch/elec/t412-asp/automix/results"

# Parse key=value pairs from command line
for arg in "$@"; do
  eval "$arg"
done

# Run Python command
python - <<EOF
import omegaconf
from inference.inference_benchmark import InferenceBenchmark

method_args = {
    "FxGenerator_code": "$FxGenerator_code",
    "FxProcessor_code": "$FxProcessor_code",
    "T": $T,
    "Schurn": $Schurn,
    "cfg_scale": $cfg_scale,
}

method_args = omegaconf.OmegaConf.create(method_args)
evaluator = InferenceBenchmark(
    method_args=method_args,
    dataset_code="$dataset",
    extra_id="$extra_id",
    path_results="$path_results"
)
evaluator.run_evaluation()
EOF
