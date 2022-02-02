
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


# one gaussian clusters

N <- 10000
mu <- c(-1);
sigma <- c(1);
y <- rnorm(N, mu, sigma);

theta_prior <- 5

stan_rdump(c("N", "y","mu", "sigma", "theta_prior"), file="out/single_gaussian.data.R")

# plot likelihood
single_gaussian <- read_rdump("out/single_gaussian.data.R")

single_gaussian_max_marginal_ll = optim(rnorm(4), 
        fn = gmm_marginal_ll, 
        method = "BFGS", 
        control = list(fnscale = -1),
        input_data = single_gaussian,
        beta_a=0.5, beta_b=0.5)

alphas = linspace(0, 1, n = 1000)
single_gaussian_ll_alphas_est = sapply(alphas, gmm_ll, params=single_gaussian_max_marginal_ll$par, input_data=single_gaussian)

single_gaussian_ll_alphas_true = sapply(alphas, gmm_ll, params=c(-0.5,0,-0.5,0), input_data=single_gaussian)

write.csv(single_gaussian_ll_alphas_est, "out/single_gaussian_ll_alphas_est.csv")
write.csv(single_gaussian_ll_alphas_true, "out/single_gaussian_ll_alphas_true.csv")
write.csv(single_gaussian_max_marginal_ll$par, "out/single_gaussian_optim_params.csv")





stan_file = 'gauss_mix.stan'
# fit by hmc
degenerate_fit_hmc <- stan(file=stan_file, data=single_gaussian,
                       chains=4, seed=483892929, refresh=2000)

# fit by vb
gauss_mix_stan = stan_model(file=stan_file, verbose=TRUE)
degenerate_fit_vbmf <- vb(gauss_mix_stan, data=single_gaussian, 
	tol_rel_obj = 1e-3, algorithm ="meanfield")

degenerate_fit_vbfr <- vb(gauss_mix_stan, data=single_gaussian, 
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

	write.csv(theta_sample, paste(c("out/single_gaussian_theta_sample_", method, ".csv"), collapse = ''))
}




	# pdf(paste(c("out/gmm_sym_prior_", method, ".pdf"), collapse = ''))

# 	if (method=="hmc"){
# 		params1 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,1,])
# 		params2 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,2,])
# 		params3 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,3,])
# 		params4 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,4,])

# 		par(mar = c(4, 4, 0.5, 0.5))
# 		plot(params1$"mu[1]", params1$"mu[2]", col=c_dark_highlight_trans, pch=16, cex=0.8,
# 		     xlab="mu1", xlim=c(-3, 3), ylab="mu2", ylim=c(-3, 3))
# 		points(params2$"mu[1]", params2$"mu[2]", col=c_dark_trans, pch=16, cex=0.8)
# 		points(params3$"mu[1]", params3$"mu[2]", col=c_mid_highlight_trans, pch=16, cex=0.8)
# 		points(params4$"mu[1]", params4$"mu[2]", col=c_mid_trans, pch=16, cex=0.8)
# 		lines(0.08*(1:100) - 4, 0.08*(1:100) - 4, col="grey", lw=2)
# 		legend("topright", c("Chain 1", "Chain 2", "Chain 3", "Chain 4"),
# 		       fill=c(c_dark_highlight_trans, c_dark_trans,
# 		              c_mid_highlight_trans, c_mid_trans), box.lty=0, inset=0.0005)
# 	}
# 	else {
# 		params1 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,1,])
# 		par(mar = c(4, 4, 0.5, 0.5))
# 		plot(params1$"mu[1]", params1$"mu[2]", col=c_dark_highlight_trans, pch=16, cex=0.8,
# 		     xlab="mu1", xlim=c(-3, 3), ylab="mu2", ylim=c(-3, 3))
# 		legend("topright", c("vb"),
# 		       fill=c(c_dark_highlight_trans), box.lty=0, inset=0.0005)
# 	}
# 	dev.off()
# }


