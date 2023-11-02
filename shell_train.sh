#!/bin/bash

# Go to the current folder
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Setup environment (adapt if necessary)
source /srv/beegfs-benderdata/scratch/${USER}/data/conda/etc/profile.d/conda.sh
conda activate py39

export DATA=/srv/beegfs-benderdata/scratch/ac_course/data/project_1/miniscapes.tar
export TMPDIR=/scratch/${USER}
export SAVEDIR=/srv/beegfs-benderdata/scratch/${USER}/data/ex1_submission

# setup weights and biases
export WANDB_API_KEY=$(cat "wandb.key")
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}

# Extract dataset (do not change this)
echo "Extract miniscapes"
mkdir -p ${TMPDIR}
mkdir -p ${SAVEDIR}
tar -xf ${DATA} -C ${TMPDIR}

# BEGIN YOUR CHANGES HERE
export TEAM_ID=20

# Run training
echo "Start training"

# You can specify the hyperparameters and the experiment name here.
python -m source.scripts.train \
  --log_dir ${SAVEDIR} \
  --dataset_root ${TMPDIR}/miniscapes \
  --name Default \
  --model_name adaptive_depth \
  --optimizer adam \
  --tasks depth semseg \
  --optimizer_lr 0.0001 \
  --batch_size 16 \
  --num_epochs 10 \
  --workers ${SLURM_CPUS_PER_TASK} \
  --workers_validation ${SLURM_CPUS_PER_TASK} \
  --batch_size_validation 16 \
  --optimizer_float_16 no \
  # ... you can pass further arguments as specified in utils/config.py
  # DO NOT FORGET ADDING BACKSLASHES for additional flags (except the last one)

# END YOUR CHANGES HERE

