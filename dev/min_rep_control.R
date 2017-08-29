#A generative model for LDA
#Imports
require(MCMCpack)
require(mvtnorm)
require(Rcpp)

set.seed(1)


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
V <- 2
M <- 100
N.mu <- 50
eta <- rep(1, V)

alpha <- rep(1, K)
ret <- gen.lda(K, V, M, N.mu, eta, alpha)
docs <- ret$docs
Ns <- ret$Ns
BETA <- ret$BETA
THETA <- ret$THETA

sourceCpp("min_rep.cpp")

i <- sapply(1:100000, function(i) {
           MinRep(docs, V)
})
