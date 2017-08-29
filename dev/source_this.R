#This script sources everything needed to run my LDA C code.
require(Rcpp)
sourceCpp("/home/nathan/Documents/Research/LdaFuncs/my_weighted_cvb0.cpp")

wLDA <- function(docs, alpha, eta, K, V, thresh, iters, seed, weights) {
    ret <- r_weighted_cvb_zero_inference(docs, alpha, eta, K, V, thresh, iters, seed, weights)
    ret$GAMMA <- t(ret$GAMMA)
    return(ret);
}

