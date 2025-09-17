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