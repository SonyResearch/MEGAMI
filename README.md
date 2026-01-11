# MEGAMI: Multitrack Embedding Generative Auto Mixing

### Automatic Music Mixing Using a Generative Model of Effect Embeddings

---

## Overview

**MEGAMI** (Multitrack Embedding Generative Auto MIxing) is a research framework for **automatic music mixing** based on a **generative model of effect embeddings**.  
Unlike conventional regression-based methods that predict a single deterministic mix, MEGAMI employs **conditional diffusion models** to capture the diversity of professional mixing decisions, modeling the inherently *one-to-many* nature of the task.

The framework operates in an **effect embedding space** rather than directly in the audio domain, enabling realistic and flexible mixing without altering musical content.  
Through **domain adaptation in the CLAP embedding space**, MEGAMI can train on both *dry* (unprocessed) and *wet* (professionally mixed) recordings. A permutation-equivariant transformer architecture allows operation on an arbitrary number of unlabeled tracks, and evaluations show performance approaching **human-level quality** across diverse musical genres.


---

## 🧩 Key Components

| Component | Description |
|------------|-------------|
| **FxGenerator** | Generates per-track effect embeddings conditioned on raw track features. |
| **FxProcessor** | Neural effects processor applying the generated embeddings to audio. |
| **CLAP Domain Adaptor** | Adapts representations between dry and wet domains using CLAP embeddings. |

---

## 🧱 Repository Structure

```

├── train_CLAPDomainAdaptor.py          # Train domain adaptation model
├── train_FxGenerator.py                # Train generative embedding model
├── train_FxProcessor.py                # Train neural effects processor
│
├── train_CLAPDomainAdaptor_public.sh   # Example public training scripts
├── train_FxGenerator_public.sh
├── train_FxProcessor_public.sh


├── inference/                          # Inference and validation modules
│   ├── inference.py
│   ├── sampler_euler_heun_multitrack.py
│
├── datasets/                           # Dataset loaders
│   ├── MoisesDB_MedleyDB_multitrack.py
│   ├── public_multidataset_singletrack.py
│   └── eval_benchmark.py
│
├── networks/                           # Network definitions
│   ├── MLP_CLAP_regressor.py
│   ├── blackbox_TCN.py
│   ├── dit_multitrack.py
│   └── transformer.py
│
├── utils/                              # Utility functions and feature extractors
│   ├── MSS_loss.py
│   ├── common_audioeffects.py
│   ├── fxencoder_plusplus/
│   ├── laion_clap/
│   ├── training_utils.py
│   └── feature_extractors/
│
├── examples/                           # Contains subdirectories of dry track set examples to run inference on.
├── conf/                               # Hydra configuration files
├── checkpoints/                        # Path where pretrained model checkpoints are expected to be 
├── run_inference.sh                    # Script for running a single song inference, using a directory containing dry tracks
├── requirements.txt                    # Dependencies
└── README.md

````

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/MEGAMI.git
   cd MEGAMI
   ```

2. **Create and activate a Conda environment**

   ```bash
   conda create -n automix python=3.13
   conda activate automix
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configuration System (Hydra)

The codebase uses [Hydra](https://hydra.cc/) for modular configuration.
Each training or inference script loads a YAML config from `conf/` and allows runtime overrides.

**Example public config (simplified):** `conf/conf_FxGenerator_Public.yaml`

```yaml
defaults:
  - dset: MoisesDB_MedleyDB_FxGenerator
  - tester: evaluate_FxGenerator
  - logging: base_logging_FxGenerator

model_dir: "experiments/example"

exp:
  exp_name: "example"
  optimizer:
    _target_: "torch.optim.AdamW"
    lr: 1e-4
  batch_size: 8
diff_params:
  type: "ve_karras"
  content_encoder_type: "CLAP"
  style_encoder_type: "FxEncoder++_DynamicFeatures"
  CLAP_args:
    ckpt_path: "checkpoints/music_audioset_epoch_15_esc_90.14.patched.pt"
```

To override parameters at runtime:

```bash
python train_FxGenerator.py model_dir=experiments/test_run exp.optimizer.lr=5e-5 exp.batch_size=16
```

---

## 🧭 Logging (Weights & Biases)

Logging is handled through **Weights & Biases (wandb)**.
By default, if `logging.log=True` in your config, a new run is created automatically and all training metrics and configurations are logged.

**To disable wandb:**

```bash
python train_FxGenerator.py logging.log=false
```

You can also change the project or entity directly in the config:

```yaml
logging:
  log: true
  wandb:
    project: "MEGAMI"
    entity: "your_wandb_username"
```

---

## 🚀 Usage

### Training

Example using the provided public scripts:

```bash
bash train_FxGenerator_public.sh
bash train_FxProcessor_public.sh
bash train_CLAPDomainAdaptor_public.sh
```

These scripts automatically create experiment directories under `experiments/` and call:

```bash
python train_FxGenerator.py --config-name=conf_FxGenerator_Public.yaml
```

Logs and checkpoints are saved under `experiments/<exp_name>/` unless otherwise specified.

### Inference

To run inference on a single track, check the script:

```bash
bash run_inference.sh
```
The script requires creating a directory containing a set of dry tracks in .wav format, sampled at 44.1 kHz. Examples are provided in "examples/". 


---

## 📊 Checkpoints

To reproduce the results reported in the MEGAMI paper, the following pretrained checkpoints are required. All files should be placed in the directory:

```
checkpoints/
```

| File name                                      | Description                                                                                                                                           |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CLAP_DA_public.pt`                            | Public CLAP-based domain adaptation checkpoint for effects removal  — [Download link](https://github.com/SonyResearch/MEGAMI/releases/download/v0/CLAP_DA_public.pt).                                                                                   |
| `FxGenerator_public.pt`                        | Public FxGenerator diffusion checkpoint operating in the embedding space — [Download link](https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxGenerator_public.pt).                                                                             |
| `FxProcessor_public.pt`                        | Public FxProcessor checkpoint (black-box TCN model) — [Download link](https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxProcessor_public.pt).  .                                                                                                  |
| `music_audioset_epoch_15_esc_90.14.patched.pt` | LAION-CLAP (music) public checkpoint — [Original link](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt). |
| `fxenc_plusplus_default.pt`                    | FXEncoder++ public checkpoint — [Original link](https://huggingface.co/yytung/fxencoder-plusplus/blob/main/fxenc_plusplus_default.pt).            |




---

## 🧾 Citation

If you use this framework in your research, please cite:

```bibtex
@article{moliner2025megami,
  title={Automatic Music Mixing Using a Generative Model of Effect Embeddings},
  author={Moliner, Eloi and Martínez-Ramírez, Marco A. and Koo, Junghyun and Liao, Wei-Hsiang and  Cheuk, Kin Wai and Serrà, Joan  and Välimäki, Vesa  and Mitsufuji, Yuki  },
  journal={Preprint},
}
```
---

## License
 
 This project is licensed under the *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License* (CC BY-NC-SA 4.0). For commercial licensing inquiries: marco [dot] martinez @ sony [dot] com
  year={2025}


