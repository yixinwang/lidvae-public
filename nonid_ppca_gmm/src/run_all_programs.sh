#!/bin/bash
#


# RCODE_SWEEP="nonoverlap_gaussian_gmm_collapse single_gaussian_gmm_collapse overlap_gaussian_gmm_collaps"
RCODE_SWEEP="oned_ppca_collapse twod_ppca_collapse twod_var_ppca_collapse overlap_gaussian_gmm_collapse nonoverlap_gaussian_gmm_collapse single_gaussian_gmm_collapse"

CODE_SUFFIX=".R"
OUT_SUFFIX=".routput"

for RCODE_SWEEPi in ${RCODE_SWEEP}; do
	export FILENAME=${RCODE_SWEEPi}${CODE_SUFFIX}
	export OUTNAME=${RCODE_SWEEPi}${OUT_SUFFIX}
	NAME=${RCODE_SWEEPi}
	echo ${NAME}
	sbatch --job-name=${NAME} run_scripts.sh
done