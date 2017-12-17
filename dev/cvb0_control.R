#A generative model for LDA

#Imports
require(MCMCpack)
require(mvtnorm)
require(Rcpp)

set.seed(123)

gen.lda <- function(K, V, M, N.mu, eta, alpha) {
    #Generate automatic priors if unspecified
    if (missing(eta)) {
        eta <- rep(1, V)
    }
    if (missing(alpha)) {
        alpha <- rep(1, K)
    }

    #Create the topics
    BETA <- rdirichlet(K, eta)
    
    #Generate doc lengths
    Ns <- rpois(M, N.mu-1)+1

    #Storage for docs
    docs <- list()
    THETA <- matrix(0, nrow = M, ncol = K)

    #Generate the actual docs
    for (m in 1:M) {
        theta <- rdirichlet(1, alpha)
        THETA[m,] <- theta
        docs[[m]] <- rep(0, Ns[m])
        for (n in 1:Ns[m]) {
            z <- rmultinom(1, 1, theta)
            w <- which(rmultinom(1, 1, t(z) %*% BETA)==1)
            docs[[m]][n] <- w
        }
    }

    return(list('docs' = docs, 'Ns' = Ns, 'BETA' = BETA, 'THETA' = THETA))
}

#Example Useage.
K <- 2
V <- 7000
M <- 20
N.mu <- 10
eta <- rep(1, V)
alpha <- rep(1, K)
ret <- gen.lda(K, V, M, N.mu, eta, alpha)
docs <- ret$docs
Ns <- ret$Ns
BETA <- ret$BETA
THETA <- ret$THETA


thresh <- 1e-8
iters <- 10000
seed <- 1234
weights <- rep(1,V)

sourceCpp('scontained_wcvb0.cpp')
result <- r_weighted_cvb_zero_inference(docs, alpha, eta, K, V, thresh, iters, seed, weights)
BETA_est <- result$BETA
BETA_est <- BETA_est / rowSums(BETA_est)
print(BETA_est)
#BETA_in <- wLDA(docs, alpha, eta, K, V, thresh, iters, seed, weights)


####Try it with real data
require(LdaFuncs)
load('keylist_crescent.RData')

#Example Useage.
K <- 10
V <- length(vocab)
eta <- rep(0.1, V)
alpha <- rep(0.5, K)
thresh <- -1
iters <- 1000
seed <- 123
weights <- rep(1, V)

result <- wLDA(keylist, alpha, eta, K, V, thresh, iters, seed, weights)
BETA_est <- result$BETA
BETA_est <- BETA_est / rowSums(BETA_est)
print(BETA_est)
result$GAMMA
