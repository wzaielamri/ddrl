#!/bin/bash
#SBATCH --job-name=ddrl_attentionMap
#SBATCH --time=100:00:00
#SBATCH --mail-type=FAIL,END
#SBTACH --mail-user=wzaielamri@techfak.uni-bielefeld.de
#SBATCH --output=logs/job-centr-%j.log
#SBATCH --nodelist=handy
#SBATCH --cpus-per-task=15

. $HOME/.bashrc
conda activate ddrl
cd $HOME/ddrl_Cheetah
module load cuda/10.1
module load tensorflow/2.2

srun --job-name=ddrl_central_Hidden16Grid --time=100:00:00 --mail-type=FAIL,END --mail-user=wzaielamri@techfak.uni-bielefeld.de  --nodelist=clumsy --cpus-per-task=10 python3.6 train_experiment_1_architecture_on_flat.py --policy_scope BipedMultiEnv_Centralized
