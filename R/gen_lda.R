#!/usr/bin/Rscript
#  gen_lda.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.16.2017

#Imports
require(Rcpp)

set.seed(123)

#' Generate documents from the weighted Latent Dirichlet Allocaiton model.
#'
#' @param K The number of topics from which to generate.
#' @param V The size of the vocabulary.
#' @param M The number of documents to generate.
#' @param N.mu The average number of words per document. Follows a Poisson(N.mu - 1) + 1 distribution (to ensure that no docs will be empty).
#' @param Pi A numeric vector of length V, the weight of each token.
#' @param alpha The hyperparameter for the Dirichlet prior associated with the document by term matrix. May be a vector of length K. If a single value is passed, a K-vector will be filled with that value.
#' @param eta The hyperparameter for the Dirichlet prior associated with the document by topic matrix. May be a vector of length V. If a single value is passed, a V-vector will be filled with that value.
#' @param V The size of the vocabuary, an integer. The vocabulary itself does not need to be passed, but will simply be the integers 1:V.
#' @param keep_z Should the topic assignments of each word be retained? Useful for debugging purposes.
#' @return A list containing docs, a list of the integer vectors, Ns, the integer vectors representing document length, BETA, the topic-by-word matrix (rows sum to one, nonnegative elements), and THETA, the document-by-topic matrix (rows sum to one, nonnegative elements).
#' @examples
#' K <- 2
#' V <- 7000
#' M <- 20
#' N.mu <- 10
#' Pi <- (1:K) / K
#' eta <- rep(1, V)
#' alpha <- rep(1, K)
#' ret <- gen.lda(K, V, M, N.mu, Pi, eta, alpha)
#' docs <- ret$docs
#' Ns <- ret$Ns
#' BETA <- ret$BETA
#' THETA <- ret$THETA
gen.lda <- function(K, V, M, N.mu, Pi, eta, alpha, keep_z = FALSE) {
    #Generate automatic priors if unspecified
    if (missing(eta)) {
        eta <- rep(1, V)
    }
    if (missing(alpha)) {
        alpha <- rep(1, K)
    }

    if (V <= 1) {
        stop("Your vocabulary had probably ought to be a little bit bigger. Set V to a positive integer greater than 1")
    }

    #Create the topics
    BETA <- matrix(sapply(1:K, function(i) rgamma(V, eta, 1)), 
                   byrow = TRUE, ncol = V)
    BETA <- BETA / rowSums(BETA)
    
    #Generate doc lengths
    Ns <- rpois(M, N.mu-1)+1

    #Storage for docs
    docs <- list()
    THETA <- matrix(0, nrow = M, ncol = K)
    if (keep_z) Z <- list()

    #Generate the actual docs
    for (m in 1:M) {
        theta <- rgamma(K, alpha, 1)
        theta <- theta / sum(theta)
        THETA[m,] <- theta
        docs[[m]] <- rep(0, Ns[m])
        if (keep_z) Z[[m]] <- rep(0, Ns[m])

        # Prepare some probabilities specific to this document due to term weighting
        tB <- apply(BETA, 2, function(col) col * theta)
        wtB <- t(apply(BETA, 1, function(row) row^Pi))
        p_w <- colSums(wtB)
        for (n in 1:Ns[m]) {
            # The standard LDA sampling method:
            #z <- rmultinom(1, 1, theta)
            #w <- which(rmultinom(1, 1, t(z) %*% BETA)==1)
            # Is equivalent to this (if weights are all 1):
            w <- which(rmultinom(1, 1, p_w) == 1)
            z <- which(rmultinom(1, 1, wtB[,w] / sum(wtB[,w])) == 1)
            docs[[m]][n] <- w
            if (keep_z) Z[[m]][n] <- z
        }
    }

    ret <- list('docs' = docs, 'Ns' = Ns, 'BETA' = BETA, 'THETA' = THETA, 'Pi' = Pi)
    if (keep_z) ret$Z <- Z
    return(ret)
}
