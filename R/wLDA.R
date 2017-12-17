#!/usr/bin/Rscript
#  wLDA.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.16.2017

#' Do inference on a Latent Dirichlet Allocation model using Collapsed Variational Bayes with unequal token weighting. This code is optimized for speed over stability.
#'
#' @param docs A list of integer vectors, one integer vector for each document. Each element in a integer vector represents a word and indicates the index of each token in the vocabulary. This index should begin at 1.
#' @param alpha The hyperparameter for the Dirichlet prior associated with the document by term matrix. May be a vector of length K. If a single value is passed, a K-vector will be filled with that value.
#' @param eta The hyperparameter for the Dirichlet prior associated with the document by topic matrix. May be a vector of length V. If a single value is passed, a V-vector will be filled with that value.
#' @param K The number of topics, an integer >= 2
#' @param V The size of the vocabuary, an integer. The vocabulary itself does not need to be passed.
#' @param thresh The threshold for stopping the iterative algorithm. If the absolute change from one iteration to the next of any element of the word-topic matrices.
#' @param iters Maximum iterations. An iteration is defined as one pass over the corpus.
#' @param seed Seed for RNG.
#' @param weights A vector of positive weights of length V, defining the importance of each word in terms of Topic Formation. 
#' @return A list with an element BETA, the topic-by-word matrix, and GAMMA, the document by topic matrix.
#' @examples
#' #Generate some documents
#' K <- 2
#' V <- 4
#' M <- 20
#' N.mu <- 100
#' eta <- rep(1, V)
#' alpha <- rep(1, K)
#'
#' ret <- gen.lda(K, V, M, N.mu, eta, alpha)
#'
#' docs <- ret$docs
#' Ns <- ret$Ns
#' BETA <- ret$BETA
#' THETA <- ret$THETA
#'
#' #Do inference, see if we got the BETA right
#' ret <- wLDA(docs, alpha, eta, K, V)
#'
#' cat('True BETA:')
#' print(BETA)
#'
#' cat('Our guess:')
#' print(ret$BETA)
wLDA <- function(docs, alpha, eta, K, V, thresh = 1e-6, iters = 100, seed, weights) {
    #if the hyperparameters are exchangable, create a vector of them.
    if (length(alpha) == 1) {
        alpha <- rep(alpha, V)
    }
    if (length(eta) == 1) {
        eta <- rep(eta, V)
    }

    #Set a seed randomly if non was specified
    if (missing(seed)) {
        seed <- sample(1:10000, 1)
    }

    #If weights are not specified, make them uniform
    if (missing(weights)) {
        weights <- rep(1, V)
    }

    #Call the underlying C code.
    ret <- r_weighted_cvb_zero_inference(docs, alpha, eta, K, V, thresh, iters, seed, weights)

    #Make BETA sum to 1
    ret$BETA <- ret$BETA / rowSums(ret$BETA)

    #Return to column-major form.
    ret$GAMMA <- t(ret$GAMMA)
    return(ret);
}
