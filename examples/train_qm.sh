#!/bin/bash -x


#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=16:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --account=atmlaml
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
source /p/project/hai_drug_qm/sc_venv_template/activate.sh

srun python train_qm.py