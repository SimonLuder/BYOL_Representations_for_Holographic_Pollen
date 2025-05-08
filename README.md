# BYOL Representations for Holographic Pollen

Repository to train a image encoder with BYOL.

## Docker

Build the docker image the with:
```
docker build -f ./dockerfile/dockerfile.slurm -t byol_training .
```

After building, start the container on any shell with:
```
docker run -it --rm -v .:/app -w /app --gpus all --env WANDB_API_KEY=$(cat wandb_api_key.secret) byol_training bash
```

