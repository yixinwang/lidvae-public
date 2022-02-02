
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

# two overlapping gaussian clusters

N <- 10000
mu <- c(-0.5, 0.5);
sigma <- c(1, 1);
lambda <- 0.15
z <- rbinom(N, 1, lambda) + 1;
y <- rnorm(N, mu[z], sigma[z]);

theta <- 0.25

theta_prior <- 5

stan_rdump(c("N", "y", "theta", "theta_prior"), file="out/overlap_gaussian.data.R")

overlap_gaussian <- read_rdump("out/overlap_gaussian.data.R")

overlap_gaussian_max_marginal_ll = optim(rnorm(4), 
        fn = gmm_marginal_ll, 
        method = "BFGS", 
        control = list(fnscale = -1),
        input_data = overlap_gaussian,
        beta_a=0.5, beta_b=0.5)

alphas = linspace(0, 1, n = 1000)
overlap_gaussian_ll_alphas_est = sapply(alphas, gmm_ll, params=overlap_gaussian_max_marginal_ll$par, input_data=overlap_gaussian)

overlap_gaussian_ll_alphas_true = sapply(alphas, gmm_ll, params=c(-0.75,0,0.75,0), input_data=overlap_gaussian)

write.csv(overlap_gaussian_ll_alphas_est, "out/overlap_gaussian_ll_alphas_est.csv")
write.csv(overlap_gaussian_ll_alphas_true, "out/overlap_gaussian_ll_alphas_true.csv")
write.csv(overlap_gaussian_max_marginal_ll$par, "out/overlap_gaussian_optim_params.csv")



stan_file = 'gauss_mix.stan'
# fit by hmc
degenerate_fit_hmc <- stan(file=stan_file, data=overlap_gaussian,
                       chains=4, seed=483892929, refresh=2000)

# fit by vb
gauss_mix_stan = stan_model(file=stan_file, verbose=TRUE)
degenerate_fit_vbmf <- vb(gauss_mix_stan, data=overlap_gaussian, 
	tol_rel_obj = 1e-3, algorithm ="meanfield")

degenerate_fit_vbfr <- vb(gauss_mix_stan, data=overlap_gaussian, 
	tol_rel_obj = 1e-3, algorithm ="fullrank")

# curate all fits
methods = c("hmc", "vbmf", "vbfr")
fits = c(degenerate_fit_hmc, degenerate_fit_vbmf, degenerate_fit_vbfr)




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
		theta_sample = params$theta
	}
	else {
		theta_sample = extract(degenerate_fit)$theta
	}

	write.csv(theta_sample, paste(c("out/overlap_gaussian_theta_sample_", method, ".csv"), collapse = ''))
}



