# Empirical studies of LIDVAE on text

This folder contains the implementation for Section 4.1 of the paper,
illustrating LIDSVAE on text datasets.

To run this study, run `src/run_fit_textidvae_sweep.sh` which fit
LIDSVAE to different datasets.

The `res/` folder contains some sample output files from fitting
LIDSVAE (after only a few epochs).

This implementation extends
https://github.com/jxhe/vae-lagging-encoder. The implementation of
LIDVAE lives in `src/modules/decoders/dec_lstm.py`.

_Note: LIDVAE tends to work well with large learning rate (e.g.
lr=0.1) and small weight decay (e.g. l2_reg=1e-6, 5e-6, 1e-5)._
