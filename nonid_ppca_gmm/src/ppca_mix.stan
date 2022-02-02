data {
  int <lower=0> N;
  int <lower=0> D;
  int <lower=0> M;
  vector[D] x[N];
}
parameters {
  matrix[M,N] z;
  matrix[D,M] w;
  real <lower=0> sigma;
  vector<lower=0>[M] alpha;
}
model {
  to_vector(z)~normal(0,1);
  for(d in 1:D) w[d]~normal(0,sigma*alpha);
  sigma~lognormal(0,1);
  alpha~inv_gamma(1,1);
  for(n in 1:N) x[n]~normal(w*col(z,n),sigma);
}

