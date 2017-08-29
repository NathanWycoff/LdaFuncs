# Minimal wrapper for cpp inference

wLDA <- function(docs, alpha, eta, K, V, thresh, iters, seed, weights) {
    ret <- r_weighted_cvb_zero_inference(docs, alpha, eta, K, V, thresh, iters, seed, weights)
    ret$GAMMA <- t(ret$GAMMA)
    return(ret);
}
