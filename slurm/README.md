# SLURM + Singularity Setup

This guide explains how to build and run the Singularity container on a SLURM-based HPC cluster. 

Note: This repository must already be set up in an environment with access to SLURM.

---

## 1. Navigate to Project Directory

```bash
cd /mnt/.../SSL_Representations_for_Holographic_Pollen
```

---

## 2. Start an Interactive SLURM Session

```bash
srun -p performance -t 60 --mem=32G --pty bash
```

* `-p performance` → partition
* `-t 60` → time (minutes)
* `--mem=32G` → memory allocation

---

## 3. Build the Singularity Container

```bash
cd repo
singularity build --fakeroot ../ssl_training.sif dockerfile/ssl_setup.def
```

This creates a `byol_training.sif` container outside the repository which is used by the setup files in the subfolders.

---

## 4. Run Trainings with the Singularity Container

Some setup scripts to run the trainings are available in the subfolders.

```
slurm/
├── byol/       # BYOL Training
├── simsiam/    # SimSiam Training
├── vicreg/     # VICReg Training
```

Example to start the training of BYOL from the respectve subfolder.
```bash
sbatch setup_byol_dual_L.sh 
```

---