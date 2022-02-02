data {
  int <lower=0> N;
  int <lower=0> D;
  int <lower=0> M;
  vector[D] x[N];
  real <lower=0> sigma;
}
parameters {
  matrix[M,N] z;
  matrix[D,M] w;
}
model {
  to_vector(z)~normal(0,1);
  for(d in 1:D) w[d]~normal(0,2);
  for(n in 1:N) x[n]~normal(w*col(z,n),sigma);
}

