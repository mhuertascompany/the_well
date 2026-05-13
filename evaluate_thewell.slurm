#!/bin/bash
#SBATCH --job-name=thewell_eval           # Nom du job
#SBATCH --output=/ltstorage/astro/abichot/THEWELL/logs/thewell_eval-out-%j.o
#SBATCH --error=/ltstorage/astro/abichot/THEWELL/logs/thewell_eval-err-%j.e
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
# Activer environnement
# ----------------------------
#module load Anaconda3/2024.02-1
#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate /ltstorage/astro/abichot/envs/thewell_py311

# ----------------------------
# Lancer l'entraînement 3D UNet
# ----------------------------
cd /ltstorage/astro/abichot/THEWELL/the_well
srun /ltstorage/astro/abichot/envs/thewell_py311/bin/python noctua/evaluate.py \
--checkpoint_dir /ltstorage/astro/abichot/THEWELL/test2/experiments \
--dataset_base /ltstorage/astro/abichot/THEWELL/the_well/datasets/sn_explosion_hr
