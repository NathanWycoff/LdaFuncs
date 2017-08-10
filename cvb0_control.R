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
V <- 4
M <- 20
N.mu <- 100
eta <- rep(1, V)
alpha <- rep(1, K)
ret <- gen.lda(K, V, M, N.mu, eta, alpha)
docs <- ret$docs
Ns <- ret$Ns
BETA <- ret$BETA
THETA <- ret$THETA

sourceCpp("cvb0.cpp")
sourceCpp("variational_lda.cpp")

thresh <- 0.001
max_iters <- 2000
#set.seed(1234)
seeds <- sample(1000, 3)
ests <- list()
i <- 1
seed <- 19

i <- 1
for (seed in seeds) {
    print(i)
    BETA_est <- RCVBZero(docs, alpha, eta, K, V, thresh, max_iters, seed)
    ests[[i]] <- BETA_est
    i <- i + 1
}

vests <- list()
i <- 1
for (seed in seeds) {
    vests[[i]] <- RVariationalEM(docs, alpha, eta[1], K, V, 0.001, 0.001, 100, 1000, seed)
    i <- i + 1
}
