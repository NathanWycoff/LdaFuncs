#ifndef CVB_H_
#define CVB_H_

double *weighted_cvb_zero_predict(int **docs, double *BETA, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in);

double **weighted_cvb_zero_inference(int **docs, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in);

#endif // CVB_H_

