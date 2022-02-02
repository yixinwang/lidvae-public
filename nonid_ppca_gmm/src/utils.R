
gmm_marginal_ll <- function(params, beta_a, beta_b, input_data, n=100){
	# alpha ~ Beta(beta_a, beta_b)
	y <- input_data$y
	alphas = rbeta(n, beta_a, beta_b)
	lls = sapply(alphas, gmm_ll, params=params, input_data=input_data)
	return(mean(lls))
}


gmm_ll <- function(alpha, params, input_data){
	mu1 = params[1]
	logsigma1 = params[2]
	mu2 = params[3]
	logsigma2 = params[4]
	y <- input_data$y
	ll <- rowLogSumExps(cbind(log(alpha)+dnorm(y, mu1, exp(logsigma1), log=TRUE), 
		log(1-alpha)+dnorm(y, mu2, exp(logsigma2), log=TRUE)))
	return(mean(ll))
}


ppca_ll <- function(z, params_w, params_logsigma, input_data){
	# x <- input_data$x[sample(nrow(input_data$x))]
	x <- rep(0, nrow(input_data$x))
	ll <- dnorm(x, t((params_w)%*%t(t(z))), exp(params_logsigma), log=TRUE)
	return(sum(ll))
}