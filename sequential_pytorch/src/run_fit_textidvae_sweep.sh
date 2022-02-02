#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

# PYCODE_SWEEP="text"
# DATASET_SWEEP="synthetic"
# # "yelp yahoo"
# MODEL_SWEEP="idvae"
# TASKID_SWEEP="2 5 8"
# LR_SWEEP="1e-1"


PYCODE_SWEEP="text"
DATASET_SWEEP="yahoo yelp"
# "yelp yahoo"
MODEL_SWEEP="idvae"
TASKID_SWEEP="2 5"
LR_SWEEP="1e-2"
# "1e-2 1e-1"




L2REG_SWEEP="5e-6"
# " 1e-6 1e-5"
OPTIM_SWEEP="rmsprop"
 # adam sgd


CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_fit_textidvae_sweep_base_emayhem.sh"

for OPTIMi in ${OPTIM_SWEEP}; do
    export OPTIM=${OPTIMi}
    for L2REGi in ${L2REG_SWEEP}; do
        export L2REG=${L2REGi}
        for LRi in ${LR_SWEEP}; do
            export LR=${LRi}
            for TASKIDi in ${TASKID_SWEEP}; do
                export TASKID=${TASKIDi}
                for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
                    NAME=${PYCODE_SWEEPi}
                    for DATASETi in ${DATASET_SWEEP}; do
                        export DATASET=${DATASETi}
                        for MODELi in ${MODEL_SWEEP}; do
                            export MODEL=${MODELi}
                            export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                            export OUTNAME=${PYCODE_SWEEPi}_data${DATASETi}_model_${MODELi}_taskid${TASKIDi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMi}_${TIMESTAMP}${OUT_SUFFIX}
                            export PRTOUT=${PYCODE_SWEEPi}_data${DATASETi}_model_${MODELi}_taskid${TASKIDi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMi}_${TIMESTAMP}${PRT_SUFFIX}
                            echo ${NAME}
                            sbatch --job-name=${NAME} --output=${OUTNAME} ${RUN_SCRIPT}
                        done
                    done
                done
            done
        done
    done
done
