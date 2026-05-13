#!/bin/bash
#SBATCH --job-name=thewell_train          # Nom du job
#SBATCH --output=/ltstorage/astro/abichot/THEWELL/logs/thewell_train-out-%j.o
#SBATCH --error=/ltstorage/astro/abichot/THEWELL/logs/thewell_train-err-%j.e
#SBATCH --partition=gpu                   # Partition GPU
#SBATCH --gres=gpu:1                      # Nombre de GPUs
#SBATCH --ntasks=1                        # Nombre de tâches
#SBATCH --cpus-per-task=4                 # Nombre de CPUs par tâche
#SBATCH --mem=128G

# ----------------------------
# Créer les dossiers de logs si nécessaire
# ----------------------------
mkdir -p /ltstorage/astro/abichot/THEWELL/logs

# ----------------------------
# Lancer l'entraînement 3D UNet
# ----------------------------
srun bash -lc "bash noctua/run_pipeline.sh"
