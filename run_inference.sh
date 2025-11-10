#!/bin/bash

# Example usage:
FxGenerator_code="public"
FxProcessor_code="public"


T=100
Schurn=10
cfg_scale=1.0
extra_id="default_run"
#path_dry_tracks="examples/3_dance"
#path_dry_tracks="examples/7_britpop"
#path_dry_tracks="examples/5_country"
#path_dry_tracks="examples/2_rock"
path_dry_tracks="examples/4_disco"
#path_dry_tracks="examples/6_grunge"

# Parse key=value pairs from command line
for arg in "$@"; do
  eval "$arg"
done

# Run Python command
python - <<EOF
import omegaconf
from inference.inference import Inference

method_args = {
    "FxGenerator_code": "$FxGenerator_code",
    "FxProcessor_code": "$FxProcessor_code",
    "T": $T,
    "Schurn": $Schurn,
    "cfg_scale": $cfg_scale,
}

method_args = omegaconf.OmegaConf.create(method_args)
evaluator = Inference(
    method_args=method_args,
)
evaluator.run_inference_single_song(directory="$path_dry_tracks", num_samples=10, exp_name="public")
EOF
