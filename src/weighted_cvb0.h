#ifndef WCVB0_H_INCLUDED
#define WCVB0_H_INCLUDED

void test(void);

double *weighted_cvb_zero_predict(int **docs, double *BETA, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in);

double **weighted_cvb_zero_inference(int **docs, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in);

#endif
