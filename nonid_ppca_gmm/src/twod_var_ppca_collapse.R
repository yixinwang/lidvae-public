
# install.packages("rstan", lib="/rigel/stats/users/yw2539/rpackages/")
# install.packages("ramify", lib="/rigel/stats/users/yw2539/rpackages/")
# install.packages("Rcpp", lib="/rigel/stats/users/yw2539/rpackages/")

.libPaths("/rigel/stats/users/yw2539/rpackages/")

options(bitmapType='cairo')

library(rstan)
library(matrixStats)
library(ramify)

rstan_options(auto_write = TRUE)

dir.create(file.path('/rigel/stats/users/yw2539/dgm_yw2539/src/neurips2020/gmm_collapse', 'out'))

set.seed(689934)

source("utils.R")

# one dimensional ppca

N <- 500
D <- 5 # output dimension
M <- 2 # latent dimension
w <- matrix(rnorm(D*M, mean=0, sd=2), D, M) 
z <- matrix(rnorm(M*N, mean=0, sd=1), M, N) 
sigma <- 1
x <- t(w%*%z) + matrix(rnorm(N*D, mean=0, sd=0.1), N, D) 


stan_rdump(c("x", "w", "z", "sigma", "N", "D", "M"), file="out/twod_ppca_var.data.R")
twod_ppca <- read_rdump("out/twod_ppca_var.data.R")

z = as.matrix(expand.grid(linspace(-2, 2, n = 50), linspace(-2, 2, n = 50)))

# logsigmas = c(-5, -4, -3, -2, -1, 0, 1)
logsigmas = c(log(0.1), log(0.2), log(0.5), log(1.), log(2))
for (logsigma in logsigmas){
	twod_ppca_ll_zs_true = apply(z, 1, ppca_ll, params_w=w, params_logsigma=logsigma, input_data=twod_ppca)

	write.csv(twod_ppca_ll_zs_true, paste(c("out/twod_ppca_ll_zs_true_logsigma",logsigma, ".csv"), collapse = ''))

	print(sd(twod_ppca_ll_zs_true))
	sigma = exp(logsigma)
	stan_rdump(c("x", "w", "z", "sigma", "N", "D", "M"), file=paste(c("out/twod_ppca_logsigma",logsigma, ".data.R"), collapse = ''))
}


for (logsigma in logsigmas){
	twod_ppca <- read_rdump(paste(c("out/twod_ppca_logsigma",logsigma, ".data.R"), collapse = ''))


	stan_file = 'ppca_knownsig.stan'
	# fit by hmc
	degenerate_fit_hmc <- stan(file=stan_file, data=twod_ppca,
	                       chains=4, seed=483892929, refresh=2000)

	# fit by vb
	# gauss_mix_stan = stan_model(file=stan_file, verbose=TRUE)
	# degenerate_fit_vbmf <- vb(gauss_mix_stan, data=twod_ppca, 
	# 	tol_rel_obj = 1e-3, algorithm ="meanfield")

	# degenerate_fit_vbfr <- vb(gauss_mix_stan, data=twod_ppca, 
	# 	tol_rel_obj = 1e-3, algorithm ="fullrank")

	# curate all fits
	# methods = c("hmc", "vbmf", "vbfr")
	# fits = c(degenerate_fit_hmc, degenerate_fit_vbmf, degenerate_fit_vbfr)
	methods = c("hmc")
	fits = c(degenerate_fit_hmc)




	# plot the fits
	c_light_trans <- c("#DCBCBCBF")
	c_light_highlight_trans <- c("#C79999BF")
	c_mid_trans <- c("#B97C7CBF")
	c_mid_highlight_trans <- c("#A25050BF")
	c_dark_trans <- c("#8F2727BF")
	c_dark_highlight_trans <- c("#7C0000BF")



	for (i in 1:length(fits)){
		method = methods[i]
		degenerate_fit = fits[i][[1]]

		print(method)
		print(degenerate_fit)

		if (method=="hmc"){
			params <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,1,])
			theta_sample = params
		}
		else {
			theta_sample = extract(degenerate_fit)
		}

		write.csv(theta_sample, paste(c("out/twod_ppca_z_sample_", method, "logsigma_", logsigma, ".csv"), collapse = ''))

	}
}


