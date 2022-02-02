# Empirical studies of latent variable non-identifiability and posterior collapse in PPCA and GMM

This folder contains the implementation for Section 4.2 of the paper,
illustrating how latent variable non-identifiability corresponds to
posterior collapse in classical probabilistic models like
Probabilistic Principal Component Analysis (PPCA) and Gaussian mixture
models (GMM).

To run this study, run `src/run_all_programs.sh` which executes
simulating and fitting PPCA and GMM under different scenarios.

The `res/` folder contains the output files and ipython notebook that
reproduces the result.

This implementation builds on https://betanalpha.github.io/assets/case_studies/identifying_mixture_models.html
