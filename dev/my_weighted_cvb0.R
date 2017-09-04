w.cvb0.inference <- function(docs, K, V, eta, alpha, weights, iters) {
    ##Collapsed Variational Bayes for weigthed LDA

    #Compute doc lengths
    Ns <- sapply(docs, length)
    M <- length(docs)

    #Initialize PHIs
    PHIS <- lapply(1:M, function(m) {
                       PHI <- matrix(0, nrow = Ns[m], ncol = K)
                       for (k in 1:K) {
                           PHI[,k] = k
                       }
                       PHI <- PHI / rowSums(PHI)
                       doc_weights <- sapply(docs[[m]], function(n) weights[n])
                       PHI <- diag(doc_weights) %*% PHI
                       return(PHI)
    })

    #Initialize Nwk and Nmk
    Nwk <- matrix(eta[1], nrow = V, ncol = K)
    for (k in 1:K) {
        for (m in 1:M) {
            for (n in 1:Ns[m]) {
                w = docs[[m]][n]
                phi = PHIS[[m]][n, k]
                Nwk[w, k] = Nwk[w, k] + phi
            }
        }
    }
    Nmk <- matrix(alpha[1], nrow = M, ncol = K)
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
    if (iters > 0) {
    for (iter in 1:iters) {
        ##Update the variational params for each word's topic assignment
        cat("\nOh shit this is a new iteration\n")
        for (m in 1:M) {
            for (n in 1:Ns[m]) {
                w <- docs[[m]][n]
                for (k in 1:K) {
                    #Remove the current val from consideration
                    Nwk[w, k] <- Nwk[w, k] - PHIS[[m]][n, k]
                    Nmk[m, k] <- Nmk[m, k] - PHIS[[m]][n, k]
                    Nk[k] <- Nk[k] - PHIS[[m]][n, k]

                    #Calculate something propto the new val
                    #print(paste("meta: ", m, ",", n, ",", k, sep = "" ))
                    first.term <- (Nwk[w, k]) / (Nk[k])
                    #print(paste("vocab_part: ", first.term))
                    second.term <- Nmk[m, k]
                    #print(paste("doc_part: ", second.term))
                    new_val <- first.term * second.term
                    PHIS[[m]][n, k] <- new_val

                    #print(paste("new_val: ", new_val,  sep = "" ))
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
    }


    BETA.mu <- t(Nwk)
    BETA.mu <- BETA.mu / rowSums(BETA.mu)
    GAMMA.mu <- Nmk / rowSums(Nmk)

    #Randomly init the assignments
    return(list("BETA" = BETA.mu, "GAMMA" = GAMMA.mu))
}
