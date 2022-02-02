#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 1
#SBATCH --time=7:59:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=rizzo,floyd,yolanda

source /proj/sml_netapp/opt/anaconda3/etc/profile.d/conda.sh

conda activate tf_gpu


echo "python ${FILENAME} -dd ${DISCRETEZ_DIM} -md ${MODEL} -cd ${CONTIZ_DIM} -od ${OUT_DIM} -opt ${OPTIMIZER} -lr ${LR} -mmt ${MOMENTUM} -ep ${NUM_EPOCHS} -l2 ${L2REG} -nfd ${NF_DIM} -nb ${NUM_BIJECTORS} -gc ${GRAD_CLIP} -nl ${NUMLAYERS} -dt ${DATASET}"

python ${FILENAME} -dd ${DISCRETEZ_DIM} -md ${MODEL} -cd ${CONTIZ_DIM} -od ${OUT_DIM} -opt ${OPTIMIZER} -lr ${LR} -mmt ${MOMENTUM} -ep ${NUM_EPOCHS} -l2 ${L2REG} -nfd ${NF_DIM} -nb ${NUM_BIJECTORS} -gc ${GRAD_CLIP} -nl ${NUMLAYERS} -dt ${DATASET}


