#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 1
#SBATCH --time=59:59:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=gonzo,floyd,yolanda

source /proj/sml_netapp/opt/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

echo "python ${FILENAME} --dataset ${DATASET} --model ${MODEL} --taskid ${TASKID} --lr ${LR} --l2reg ${L2REG} --optim ${OPTIM}"

python ${FILENAME} --dataset ${DATASET} --model ${MODEL} --taskid ${TASKID} --lr ${LR} --l2reg ${L2REG} --optim ${OPTIM}

