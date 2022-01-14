#!/bin/bash
#SBATCH --job-name=ss-paramsweep    # create a short name for your job
#SBATCH --output=ss-paramsweep_log.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
# #SBATCH --mem-per-cpu=16G         # cpu memory
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=200:00:00          # total run time limit (HH:MM:SS)

source /apps/local/conda_init.sh
conda activate ss-models

python paramsweep.py
