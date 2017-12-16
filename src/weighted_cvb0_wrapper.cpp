/*
 * An Rcpp wrapper for weighted_cvb0.c
 *
 * @author Nathan Wycoff
 * @since August 11 2017
 */

#include <Rcpp.h>
extern "C" {                                                                     
#include "weighted_cvb0.h"                                                             
}                                                                                
using namespace Rcpp;

/*
 * Function: RCVBZero
 * ----------------------------------------------
 * An R function which wraps CVBZero, does inference on LDA using collapsed variaitonal bayes.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * alpha_in: A double array containing the topic prevalence prior params
 *
 * eta_in: A double array containing the word prevalence prior params
 *
 * K: The number of topics
 *
 * V: The size of the vocab
 *
 * thresh: The threshold for convergence; if the maximal abs difference among all variational params is below this on one iteration, the algorithm will converge successfully.
 *
 * max_iters: The maximum allowable iterations.
 * 
 * seed: The seed for random number generation.
 */
// [[Rcpp::export]]
List r_weighted_cvb_zero_inference(List docs_in, NumericVector alpha_in, NumericVector eta_in, int K, int V, double thresh, int max_iters, int seed, NumericVector weights_in) {
    //Turn the inputs into something consumeable by C
    //Store the hyperparams
    double *alpha = alpha_in.begin();
    double *eta = eta_in.begin();
    double *weights = weights_in.begin();

    //Store the documents and their sizes.
    int M = docs_in.size();
    // The inplace method is producing a strange bug at present, but this is quicker.
    //IntegerVector current;
    //int **docs = (int **)malloc(M * sizeof(int *)); 
    //int *Ns = (int *) malloc(M * sizeof(int));
    //for (int m = 0; m < M; m++) {
    //    current = docs_in(m);
    //    *(docs + m) = (int *)current.begin();
    //    *(Ns + m) = (int)current.size();
    //}
    //Copy the docs to avoid the world's weirdest bug.
    int **docs = (int **)malloc(M * sizeof(int *));
    IntegerVector current;
    int *Ns = (int *)malloc(M * sizeof(int));
    int *doc;
    for (int m = 0; m < M; m++) {
        current = docs_in(m);
        *(Ns + m) = current.size();
        doc = (int *)malloc(*(Ns + m) * sizeof(int));
        for (int n = 0; n < *(Ns + m); n++) {
            *(doc + n) = current(n);
        }
        *(docs + m) = doc;
    }

    //Run the actual C script; unpack the results
    double **c_ret = weighted_cvb_zero_inference(docs, Ns, alpha, eta, K, V, M, thresh, max_iters, seed,weights);
    double *Nwk = *(c_ret);
    double *Nmk = *(c_ret + 1);

    //Create the return object.
    //Note this automatically transposes the matrices.
    List ret = List::create(Named("BETA") = NumericMatrix(K, V, Nwk), Named("GAMMA") = NumericMatrix(K, M, Nmk));

    return ret;
}

/*
 * Function: RCVBZero
 * ----------------------------------------------
 * An R function which wraps CVBZero, does inference on LDA using collapsed variaitonal bayes.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * alpha_in: A double array containing the topic prevalence prior params
 *
 * eta_in: A double array containing the word prevalence prior params
 *
 * K: The number of topics
 *
 * V: The size of the vocab
 *
 * thresh: The threshold for convergence; if the maximal abs difference among all variational params is below this on one iteration, the algorithm will converge successfully.
 *
 * max_iters: The maximum allowable iterations.
 * 
 * seed: The seed for random number generation.
 */
// [[Rcpp::export]]
NumericMatrix r_weighted_cvb_zero_predict(List docs_in, NumericMatrix BETA_in, NumericVector alpha_in, NumericVector eta_in, int K, int V, double thresh, int max_iters, int seed, NumericVector weights_in) {
    //Turn the inputs into something consumeable by C
    //Store the hyperparams
    double *alpha = alpha_in.begin();
    double *eta = eta_in.begin();
    double *weights = weights_in.begin();
    double *BETA = BETA_in.begin();

    //Store the documents and their sizes.
    int M = docs_in.size();
    // The inplace method is producing a strange bug at present, but this is quicker.
    //IntegerVector current;
    //int **docs = (int **)malloc(M * sizeof(int *)); 
    //int *Ns = (int *) malloc(M * sizeof(int));
    //for (int m = 0; m < M; m++) {
    //    current = docs_in(m);
    //    *(docs + m) = (int *)current.begin();
    //    *(Ns + m) = (int)current.size();
    //}
    //Copy the docs to avoid the world's weirdest bug.
    int **docs = (int **)malloc(M * sizeof(int *));
    IntegerVector current;
    int *Ns = (int *)malloc(M * sizeof(int));
    int *doc;
    for (int m = 0; m < M; m++) {
        current = docs_in(m);
        *(Ns + m) = current.size();
        doc = (int *)malloc(*(Ns + m) * sizeof(int));
        for (int n = 0; n < *(Ns + m); n++) {
            *(doc + n) = current(n);
        }
        *(docs + m) = doc;
    }

    //Run the actual C script.
    double *Nmk = weighted_cvb_zero_predict(docs, BETA, Ns, alpha, eta, K, V, M, thresh, max_iters, seed,weights);

    //Note this automatically transposes the matrix, as desired.
    return NumericMatrix(K, M, Nmk);
}
