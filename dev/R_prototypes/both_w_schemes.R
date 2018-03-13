#!/usr/bin/Rscripj
#  my_weighted_cvb0.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.21.2018

#'
#' @param docs
#' @param K
#' @param V
#' @param eta
#' @param alpha
#' @param weights
#' @param iters
#' @param wilchew Use the parameterization from Wilson and Chew?
w.cvb0.inference <- function(docs, K, V, eta, alpha, weights, iters, willchew = FALSE) {
    ##Collapsed Variational Bayes for weigthed LDA

    #Compute doc lengths
    Ns <- sapply(docs, length)
    M <- length(docs)

    #Initialize PHIs
    PHIS <- lapply(1:M, function(m) {
                       PHI <- matrix(runif(Ns[m] * K), nrow = Ns[m], ncol = K)
                       PHI <- PHI / rowSums(PHI)
                       doc_weights <- sapply(docs[[m]], function(n) weights[n])
                       PHI <- diag(doc_weights) %*% PHI
                       return(PHI)
    })

    #Initialize Nwk and Nmk
    Nwk <- matrix(0, nrow = V, ncol = K)
    for (k in 1:K) {
        for (m in 1:M) {
            for (n in 1:Ns[m]) {
                w = docs[[m]][n]
                phi = PHIS[[m]][n, k]
                Nwk[w, k] = Nwk[w, k] + phi
            }
        }
    }
    Nmk <- matrix(0, nrow = M, ncol = K)
    for (m in 1:M) {
        for (k in 1:K) {
            for (n in 1:Ns[m]) {
                phi = PHIS[[m]][n, k]
                Nmk[m, k] = Nmk[m, k] + phi
            }
        }
    }

    #Initiliaze Nk
    Nk <- rep(0, K)
    for (k in 1:K) {
        for (m in 1:M) {
            Nk[k] <- Nk[k] + Nmk[m, k]
        }
    }
    
    BETA.mu <- matrix(0, nrow = K, ncol = V)
    
    #Do the iters!
    for (iter in 1:iters) {
        ##Update the variational params for each word's topic assignment
        for (m in 1:M) {
            for (n in 1:Ns[m]) {
                w <- docs[[m]][n]
                for (k in 1:K) {
                    #Remove the current val from consideration
                    Nwk[w, k] <- Nwk[w, k] - PHIS[[m]][n, k]
                    Nmk[m, k] <- Nmk[m, k] - PHIS[[m]][n, k]
                    Nk[k] <- Nk[k] - PHIS[[m]][n, k]

                    #Calculate something propto the new val
                    first.term <- (Nwk[w, k] + eta[w]) / (Nk[k] + sum(eta))
                    second.term <- Nmk[m, k] + alpha[k]
                    PHIS[[m]][n, k] <- first.term * second.term
                }
                #Normalize PHI
                PHIS[[m]][n,] <- weights[w] * PHIS[[m]][n,] / sum(PHIS[[m]][n,])


                #Update the counts
                for (k in 1:K) {
                    Nwk[w, k] <- Nwk[w, k] + PHIS[[m]][n, k]
                    Nmk[m, k] <- Nmk[m, k] + PHIS[[m]][n, k]
                    Nk[k] <- Nk[k] + PHIS[[m]][n, k]
                }
            }
        }

    }

    #If willson and chew parameterization is desired, we need to recalculate Nwk with all PHIS treated equally.
    if (willchew) {
        for (PHI in PHIS) {
            for (n in nrow(PHI)) {
                PHI[n,] <- PHI[n,] / sum(PHI[n,])
            }
        }
    }


    norm.Nwk <- Nwk / colSums(Nwk)
    BETA.mu <- t(norm.Nwk)
    GAMMA.mu <- Nmk / rowSums(Nmk)

    #Randomly init the assignments
    return(list("BETA" = BETA.mu, "GAMMA" = GAMMA.mu))
}
