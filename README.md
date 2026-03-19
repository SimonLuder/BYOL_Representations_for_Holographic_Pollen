# Self-Supervised Representation Learning for Holographic Pollen

This repository provides a PyTorch-based training pipeline for learning self-supervised image representations on holographic pollen data. The goal is to train image encoders that can generate meaningful representations without requiring explicit labels.

The following self-supervised learning methods are currently implemented:

- Bootstrap Your Own Latents (BYOL) \[ [arXiv](https://arxiv.org/abs/2006.07733) \]
- Simple Siamese Representation Learning (SimSiam) \[ [arXiv](https://arxiv.org/abs/2011.10566) \]
- Variance Invariance Covarianve Regularization (VICReg) \[ [arXiv](https://arxiv.org/abs/2105.04906) \]


------------
## Structure
```
├── config/                     # Configuration files & hyperparameters
├── dockerfile/                 # Docker build recipes (docker and singularity images)
├── ssl_poleno/                 # Core self-supervised code
├── tools/                      # Main executable scripts
├── requirements.txt            # Python dependencies
├── pyproject.toml
└── README.md
```
------------

## Usage

The training can be run locally:

```powershell
# BYOL
python -m tools.train_lightning --config config/base_byol_dual_config.yaml

# SimSiam
python -m tools.train_lightning --config config/base_simsiam_dual_config.yaml

# VICReg
python -m tools.train_lightning --config config/base_vicreg_dual_config.yaml
```

## Setup

### Option 1: Local (pip)

```bash
pip install -r requirements.txt
```

## Option 2: Docker

Build the docker image the with:
```bash
docker build -f ./dockerfile/dockerfile.slurm -t ssl_training .
```

After building, start the container on any shell with:
```bash
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  --gpus all \
  --env-file .env \
  ssl_training bash
```

### Option 3: SLURM + Singularity

```bash
cd /path/to/repo
srun -p performance -t 60 --mem=32G --pty bash
singularity build --fakeroot ../ssl_training.sif dockerfile/ssl_setup.def
```

For full instructions, see [SLURM Setup](slurm/README.md)
