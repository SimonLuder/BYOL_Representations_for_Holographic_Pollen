# Self-Supervised Representations for Holographic Pollen

Repository to train self-supervised image encoders for holographic pollen data. This repository implements a PyTorch based training pipeline for self-supervised learning of image encoders to create image labels without explicit supervision.

The following methods are currently supported:

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

```powershell
# BYOL
python -m tools.train_lightning --config config/base_byol_dual_config.yaml

# SimSiam
python -m tools.train_lightning --config config/base_simsiam_dual_config.yaml

# VICReg
python -m tools.train_lightning --config config/base_vicreg_dual_config.yaml
```

## Docker

Build the docker image the with:
```
docker build -f ./dockerfile/dockerfile.slurm -t byol_training .
```

After building, start the container on any shell with:
```
docker run -it --rm -v .:/app -w /app --gpus all --env WANDB_API_KEY=$(cat wandb_api_key.secret) byol_training bash
```

## SLURM
Create tar file for export to slurm
```
docker save byol_training -o byol_training.tar
```

Start interative session
```
srun -p performance -t 60 --pty bash
```

Create singularity container
```
singularity build --fakeroot --sandbox byol_sandbox docker-archive://byol_training.tar
```