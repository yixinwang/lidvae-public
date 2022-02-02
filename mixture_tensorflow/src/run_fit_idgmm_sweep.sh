#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="image"
# MODEL_SWEEP="idvae vae laggingvae betavae li_et_al"
MODEL_SWEEP="idvae"
CONTIZ_DIM_SWEEP="64"
OUT_DIM_SWEEP="784"
OPTIMIZER_SWEEP="rmsprop" 
LR_SWEEP="1e-3" 
MOMENTUM_SWEEP="0.5" 
NUM_EPOCHS_SWEEP="20000"
L2REG_SWEEP="1e-4"
NF_DIM_SWEEP="128"
NUM_BIJECTORS_SWEEP="2"
GRAD_CLIP_SWEEP="5"
NUMLAYERS_SWEEP="2"
# DATASET_SWEEP="omniglot mnist fashionmnist pinwheel"

DATASET_SWEEP="mnist fashionmnist"
DISCRETEZ_DIM_SWEEP="10"

# DATASET_SWEEP="pinwheel"
# DISCRETEZ_DIM_SWEEP="5"

# DATASET_SWEEP="omniglot"
# DISCRETEZ_DIM_SWEEP="50"


CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_fit_idgmm_sweep_base_emayhem.sh"


for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
	export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
	export OUTNAME=${PYCODE_SWEEPi}_${TIMESTAMP}${OUT_SUFFIX}
	export PRTOUT=${PYCODE_SWEEPi}_${TIMESTAMP}${PRT_SUFFIX}
	NAME=${PYCODE_SWEEPi}
	for DISCRETEZ_DIMi in ${DISCRETEZ_DIM_SWEEP}; do
		export DISCRETEZ_DIM=${DISCRETEZ_DIMi}
		for MODELi in ${MODEL_SWEEP}; do
			export MODEL=${MODELi}
			for CONTIZ_DIMi in ${CONTIZ_DIM_SWEEP}; do
				export CONTIZ_DIM=${CONTIZ_DIMi}
				for OUT_DIMi in ${OUT_DIM_SWEEP}; do
					export OUT_DIM=${OUT_DIMi}
					for OPTIMIZERi in ${OPTIMIZER_SWEEP}; do
						export OPTIMIZER=${OPTIMIZERi}
						for LRi in ${LR_SWEEP}; do
							export LR=${LRi}
							for MOMENTUMi in ${MOMENTUM_SWEEP}; do
								export MOMENTUM=${MOMENTUMi}
								for NUM_EPOCHSi in ${NUM_EPOCHS_SWEEP}; do
									export NUM_EPOCHS=${NUM_EPOCHSi}
									for L2REGi in ${L2REG_SWEEP}; do
										export L2REG=${L2REGi}
										for NF_DIMi in ${NF_DIM_SWEEP}; do
											export NF_DIM=${NF_DIMi}
											for NUM_BIJECTORSi in ${NUM_BIJECTORS_SWEEP}; do
												export NUM_BIJECTORS=${NUM_BIJECTORSi}
												for GRAD_CLIPi in ${GRAD_CLIP_SWEEP}; do
													export GRAD_CLIP=${GRAD_CLIPi}
													for NUMLAYERSi in ${NUMLAYERS_SWEEP}; do
														export NUMLAYERS=${NUMLAYERSi}		
														for DATASETi in ${DATASET_SWEEP}; do
															export DATASET=${DATASETi}		
                            								export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                            								export OUTNAME=${PYCODE_SWEEPi}_data${DATASETi}_model_${MODELi}_NUMLAYERS${NUMLAYERSi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMIZERi}_${TIMESTAMP}${OUT_SUFFIX}
								                            export PRTOUT=${PYCODE_SWEEPi}_data${DATASETi}_model_${MODELi}_taskid${NUMLAYERSi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMIZERi}_${TIMESTAMP}${PRT_SUFFIX}
								                            echo ${NAME}
								                            sbatch --job-name=${NAME} --output=${OUTNAME} ${RUN_SCRIPT}
														done
													done
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done
