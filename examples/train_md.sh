#!/bin/bash -x


#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=23:59:59
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --account=atmlaml
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
source /p/project/hai_drug_qm/sc_venv_template/activate.sh

srun python train_md.py