// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// r_weighted_cvb_zero_inference
List r_weighted_cvb_zero_inference(List docs_in, NumericVector alpha_in, NumericVector eta_in, int K, int V, double thresh, int max_iters, int seed, NumericVector weights_in);
RcppExport SEXP _LdaFuncs_r_weighted_cvb_zero_inference(SEXP docs_inSEXP, SEXP alpha_inSEXP, SEXP eta_inSEXP, SEXP KSEXP, SEXP VSEXP, SEXP threshSEXP, SEXP max_itersSEXP, SEXP seedSEXP, SEXP weights_inSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type docs_in(docs_inSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alpha_in(alpha_inSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type eta_in(eta_inSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type V(VSEXP);
    Rcpp::traits::input_parameter< double >::type thresh(threshSEXP);
    Rcpp::traits::input_parameter< int >::type max_iters(max_itersSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type weights_in(weights_inSEXP);
    rcpp_result_gen = Rcpp::wrap(r_weighted_cvb_zero_inference(docs_in, alpha_in, eta_in, K, V, thresh, max_iters, seed, weights_in));
    return rcpp_result_gen;
END_RCPP
}
// r_weighted_cvb_zero_predict
NumericMatrix r_weighted_cvb_zero_predict(List docs_in, NumericMatrix BETA_in, NumericVector alpha_in, NumericVector eta_in, int K, int V, double thresh, int max_iters, int seed, NumericVector weights_in);
RcppExport SEXP _LdaFuncs_r_weighted_cvb_zero_predict(SEXP docs_inSEXP, SEXP BETA_inSEXP, SEXP alpha_inSEXP, SEXP eta_inSEXP, SEXP KSEXP, SEXP VSEXP, SEXP threshSEXP, SEXP max_itersSEXP, SEXP seedSEXP, SEXP weights_inSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type docs_in(docs_inSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type BETA_in(BETA_inSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alpha_in(alpha_inSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type eta_in(eta_inSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type V(VSEXP);
    Rcpp::traits::input_parameter< double >::type thresh(threshSEXP);
    Rcpp::traits::input_parameter< int >::type max_iters(max_itersSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type weights_in(weights_inSEXP);
    rcpp_result_gen = Rcpp::wrap(r_weighted_cvb_zero_predict(docs_in, BETA_in, alpha_in, eta_in, K, V, thresh, max_iters, seed, weights_in));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_LdaFuncs_r_weighted_cvb_zero_inference", (DL_FUNC) &_LdaFuncs_r_weighted_cvb_zero_inference, 9},
    {"_LdaFuncs_r_weighted_cvb_zero_predict", (DL_FUNC) &_LdaFuncs_r_weighted_cvb_zero_predict, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_LdaFuncs(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
