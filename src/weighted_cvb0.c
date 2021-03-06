#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>


/*
 * Function: RandUnif
 * ----------------------------------------------
 * Generate a random number between 0 and 1 (hopefully pretty uniform).
 *
 * returns: A double between 0 and 1.
 */
double RandUnif() {
    return ((double)rand()/(double)RAND_MAX);
}


/* Function: LinfDistance
 * ----------------------------------------------
 * Get l_{infty} norm (max absolute difference) between two vectors of length N
 */
double LinfDistance(double *vec1, double *vec2, int N) {
    double max = 0;
    double absdiff;
    double val1, val2;
    for (int n = 0; n < N; n++) {
        val1 = *(vec1 + n);
        val2 = *(vec2 + n);
        absdiff = abs(val1 - val2);
        if (absdiff > max) {
            max = absdiff;
        }
    }
    return max;
}

/*
 * Function: InitPHIS
 * ----------------------------------------------
 * Initialize, for each document, a PHI, the variational parameter for term-topic belongings by drawing values from uniform(0,1) then normalizing.
 *
 * Ns: An array of length M, giving the length of each document.
 *
 * M: The number of docs || length of Ns
 *
 * K: The number of topics
 *
 * returns: A pointer to a bunch of NxK arrays, where N is the number of words in each document (row-major format).
 */
double **InitPHIS(int **docs, int *Ns, int M, int K, double *weights) {
    //Our list of matrices
    double **PHIS = (double **)malloc(M * sizeof(double *));

    //Some params for the inner loop
    int N, w;
    double *PHI;
    int *doc;
    double row_sum, r_num;

    //Create matrices with random elements whose rows sum to 1.
    for (int m = 0; m < M; m++) {
        N = *(Ns + m);
        PHI = (double *)malloc(N * K * sizeof(double));
        doc = *(docs + m);
        for (int n = 0; n < N; n++) {
            row_sum = 0.0;
            w = *(doc + n) - 1;
            //Draw some uniform values
            for (int k = 0; k < K; k++) {
                r_num = k + 1;
                row_sum += r_num;
                *(PHI + n*K + k) = r_num;
            }

            //Normalize Them
            for (int k = 0; k < K; k++) {
                *(PHI + n*K + k) *= *(weights + w) / row_sum;
            }
        }

        //Add PHI to our list
        *(PHIS + m) = PHI;
    }

    return PHIS;
}

/*
 * Function: InitNwk
 * ----------------------------------------------
 * Initialize Nwk, the expected word count by topic matrix, based on initial PHI vals. **Slightly different from Nwk in the literature in that this has the prior built in.**
 *
 * PHIS: A list of row-major matrices, the variational params for word-topic assignment.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * Ns: An array containing the length of each doc
 *
 * eta: the prior vector on each word
 *
 * M: The number of docs
 *
 * K: The number of topics
 *
 * V: The number of unique words
 *
 * Returns: A pointer to the Nwk matrix.
 */
double *InitNwk(double **PHIS, int **docs, int *Ns, double * eta, int M, int K, int V) {
    //Init Nwk to the prior
    double *Nwk = (double *)malloc(V * K * sizeof(double));
    for (int v = 0; v < V; v++) {
        for (int k = 0; k < K; k++) {
            *(Nwk + v*K + k) = *(eta + v);
        }
    }

    //Sum over PHI
    int N, w;
    double *PHI;
    int *doc;
    for (int m = 0; m < M; m++) {
        N = *(Ns + m);
        PHI = *(PHIS + m);
        doc = *(docs + m);
        //Add in the vals for the current doc.
        for (int n = 0; n < N; n++) {
            w = *(doc + n) - 1;
            for (int k = 0; k < K; k++) {
                *(Nwk + w*K + k) += *(PHI + n*K + k);
            }
        }
    }

    return Nwk;
}

/*
 * Function: InitNmk
 * ----------------------------------------------
 * Initialize Nmk, the expected topic count per document. **NOTE: unlike is typical in the literature, the prior is included in the expectation**
 *
 * PHIS: A list of row-major matrices, the variational params for word-topic assignment.
 *
 * Ns: An array containing the length of each doc
 *
 * alpha: The prior vector for each topic
 *
 * M: The number of docs
 *
 * K: The number of topics
 *
 * Returns: A pointer to the Nwk matrix.
 */
double *InitNmk(double **PHIS, int *Ns, double * alpha, int M, int K) {
    //Initialize Nmk to be the prior
    double *Nmk = (double *)malloc(M * K * sizeof(double));
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            *(Nmk + m * K + k) = *(alpha + k);
        }
    }

    //Update the values from PHI
    int N;
    double *PHI;
    for (int m = 0; m < M; m++) {
        PHI = *(PHIS + m);
        N = *(Ns + m);
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                *(Nmk + m*K + k) += *(PHI + n*K + k);
            }
        }
    }

    return Nmk;
}

/*
 * Function: InitNk
 * ----------------------------------------------
 * Initialize Nk, the array of expected counts for the entire corpus for each topic
 *
 * Nmk: The matrix of document by topic expected counts
 *
 * M: The number of docs
 *
 * K: The number of topics
 *
 * Returns: A pointer to the Nk matrix.
 */
double *InitNk(double *Nmk, int M, int K) {
    //Sum over docs to get corpus wide topic popularity.
    double *Nk = (double *)malloc(K * sizeof(double));
    for (int k = 0; k < K; k++) {
        *(Nk + k) = 0;
        for (int m = 0; m < M; m++) {
            *(Nk + k) += *(Nmk + m*K + k);
        }
    }

    return Nk;
}

/*
 * Function: DoCollapsedStep
 * ----------------------------------------------
 * Do a step of collapsed variational bayes approximated by the zeroeth order taylor expansion, updating Nwk, Nmk, Nk, and PHIS as appropriate.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * Nwk: the expected word count by topic matrix OR a row-normalized version of this if do_inference is false.
 *
 * Nmk: the expected document by topic matrix
 *
 * Nk: the expected topic prevalence matrix
 *
 * PHIS: A list of row-major matrices, the variational params for word-topic assignment.
 *
 * Ns: An array of length M, giving the length of each document.
 *
 * M: The number of docs
 *
 * K: The number of topics
 *
 * V: The size of the vocab
 *
 * do_inference: Should we update Nwk? Set to true if so. Set to false if an out of sample test is desired.
 *
 * Returns: The maximum change in this iteration for convergence evaluation purposes.
 */
double DoCollapsedStep(int **docs, double *Nwk, double *Nmk, double *Nk, double **PHIS, int *Ns, int M, int K, int V, double *weights, bool do_inference) {
    int N, w;
    double phi, vocab_part, doc_part, row_sum, new_val, change;
    double *phis_old = (double *)malloc(K * sizeof(double));//To store the old values of PHI on each iter
    double *PHI;
    int *doc;
    double max_change = 0.0;
    for (int m = 0; m < M; m++) {
        N = *(Ns + m);
        PHI = *(PHIS + m);
        doc = *(docs + m);
        for (int n = 0; n < N; n++) {
            w = *(doc + n) - 1;
            row_sum = 0.0;
            for (int k = 0; k < K; k++) {
                //Decrement the current word
                phi = *(PHI + n*K + k);
                if (do_inference) {
                    *(Nwk + w*K + k) -= phi;
                }
                *(Nmk + m*K + k) -= phi;
                *(Nk + k) -= phi;
                *(phis_old + k) = phi;

                //Calculate something propto the expected topic assignments
                if (do_inference) {
                vocab_part = *(Nwk + w * K + k) / *(Nk + k);
                } else {
                    vocab_part = *(Nwk + w * K + k);
                }

                //printf("%s%i,%i,%i\n", "meta (m,n,k): ", m, n, k);
                doc_part = *(Nmk + m*K + k);
                new_val = vocab_part * doc_part;

                //printf("%s%f\n", "doc_part: ", doc_part);
                //printf("%s%f\n", "vocab_part: ", vocab_part);

                *(PHI + n*K + k) = new_val;
                row_sum += new_val;

                //printf("%s%f\n", "new_val: ", new_val);
            }

            //Normalize PHIs
            for (int k = 0; k < K; k++) {
                *(PHI + n*K + k) *= *(weights + w) / row_sum;
            }

            //Increment the new vals for the current word.
            for (int k = 0; k < K; k++) {
                phi = *(PHI + n*K + k);
                if (do_inference) {
                    *(Nwk + w*K + k) += phi;
                }
                *(Nmk + m*K + k) += phi;
                *(Nk + k) += phi;
            }
            
            //Get the changes for this iteration
            for (int k = 0; k < K; k++) {
                change = abs(*(phis_old + k) - *(PHI + n*K + k));
                if (change > max_change) {
                    max_change = change;
                }
            }
        }
    }

    return max_change;
}


/*
 * Function: weighted_cvb_zero_inference
 * ----------------------------------------------
 * Estimate an LDA model using the Collapsed Variational Bayes Zero algorithm, a zeroth order taylor approximation of the collapsed variational bayes algorithm, very similar to collapsed gibbs sampling.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * Ns: An array of length M, giving the length of each document.
 *
 * alpha: A double array containing the topic prevalence prior params
 *
 * eta: A double array containing the word prevalence prior params
 *
 * K: The number of topics
 *
 * V: The size of the vocab
 *
 * M: The number of docs
 *
 * thresh: The threshold for convergence; if the maximal abs difference among all variational params is below this on one iteration, the algorithm will converge successfully.
 *
 * max_iters: The maximum allowable iterations.
 * 
 * seed: The seed for random number generation.
 *
 * Returns: A pointer to two pointers: The first is Nwk, the second is Nmk
 */
double **weighted_cvb_zero_inference(int **docs, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in) {
    srand(seed);
    //Normalize weights to sum to V, creating a copy so we don't spook the user.
    double sum = 0.0;
    for (int v = 0; v < V; v++) {
        sum += *(weights_in + v);
    }
    double *weights = (double *)malloc(V * sizeof(double));
    for (int v = 0; v < V; v++) {
        *(weights + v) = *(weights_in + v) / sum * (double)V;
    }

    //Randomly init the word-topic assignments
    double **PHIS = InitPHIS(docs, Ns, M, K, weights);

    //Initialize our counts based on PHI
    double *Nwk = InitNwk(PHIS, docs, Ns, eta, M, K, V);
    double *Nmk = InitNmk(PHIS, Ns, alpha, M, K);
    double *Nk = InitNk(Nmk, M, K);

    //Create a copy so we can check what for convergence
    double *old_Nwk = (double *)malloc(K*V*sizeof(double));
    for (int k = 0; k < K; k++) {
        for (int v = 0; v < V; v++) {
            *(old_Nwk + v*K + k) = *(Nwk + v*K + k);
        }
    }

    int iter = 0;
    double diff = DBL_MAX;
    double *row_sums = (double *)malloc(K * sizeof(double));//For convergence purposes
    double *old_row_sums = (double *)malloc(K * sizeof(double));//For convergence purposes
    double new_val, old_val, current_diff;
    while (iter < max_iters && diff > thresh) {
        iter += 1;
        diff = DoCollapsedStep(docs, Nwk, Nmk, Nk, PHIS, Ns, M, K, V, weights, true);
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            for (int v = 0; v < V; v++) {
                sum += *(Nwk + v*K + k);
            }
        }

        //Figure out the normalizing constants so we can check convergence.
        for (int k = 0; k < K; k++) {
            //Make row sums
            *(row_sums + k) = 0.0;
            for (int v = 0; v < V; v++) {
                *(row_sums + k) += *(Nwk + v*K + k);
            }
        }
        //Figure out the normalizing constants so we can check convergence, this time for the old Nwk
        for (int k = 0; k < K; k++) {
            //Make row sums
            *(old_row_sums + k) = 0.0;
            for (int v = 0; v < V; v++) {
                *(old_row_sums + k) += *(old_Nwk + v*K + k);
            }
        }

        //Check what the biggest change was in the topic-word matrix
        diff = 0.0;
        for (int k = 0; k < K; k++) {
            for (int v = 0; v < V; v++) {
                new_val = *(Nwk + v*K + k) / *(row_sums + k);
                old_val = *(old_Nwk + v*K + k) / *(old_row_sums + k);
                current_diff = fabs(new_val - old_val);
                if (current_diff > diff) {
                    diff = current_diff;
                }
            }
        }

        //Store the new vals in the old
        for (int k = 0; k < K; k++) {
            for (int v = 0; v < V; v++) {
                *(old_Nwk + v*K + k) = *(Nwk + v*K + k);
            }
        }
    }

    if (iter == max_iters) {
        printf("WARN: Convergence Failure in CVBZero -- Reached max_iters");
    }

    //Normalize Nmk so it becomes GAMMA
    double col_sum;
    for (int m = 0; m < M; m++) {
        //Make col sums
        col_sum = 0.0;
        for (int k = 0; k < K; k++) {
            col_sum += *(Nmk + m*K + k);
        }

        //Normalize
        for (int k = 0; k < K; k++) {
            *(Nmk + m*K + k) /= col_sum;
        }
    }


    //Free memory
    free(Nk);
    for (int m = 0; m < M; m++) {
        free(*(PHIS + m));
    }
    free(PHIS);
    free(docs);
    free(Ns);

    double **ret = (double **)malloc(2*sizeof(double *));
    
    *(ret) = Nwk;
    *(ret + 1) = Nmk;
    
    return ret;

}

/*
 * Function: weighted_cvb_zero_predict
 * ----------------------------------------------
 * Given a word-topic matrix, predict topic assignments of new documents.
 *
 * docs: A list of integer arrays, the indices in the vocab for the words in each doc
 *
 * Ns: An array of length M, giving the length of each document.
 *
 * alpha: A double array containing the topic prevalence prior params
 *
 * eta: A double array containing the word prevalence prior params
 *
 * K: The number of topics
 *
 * V: The size of the vocab
 *
 * M: The number of docs
 *
 * thresh: The threshold for convergence; if the maximal abs difference among all variational params is below this on one iteration, the algorithm will converge successfully.
 *
 * max_iters: The maximum allowable iterations.
 * 
 * seed: The seed for random number generation.
 */
double *weighted_cvb_zero_predict(int **docs, double *BETA, int *Ns, double *alpha, double *eta, int K, int V, int M, double thresh, int max_iters, int seed, double *weights_in) {
    srand(seed);
    //Normalize weights to sum to V, creating a copy so we don't spook the user.
    double sum = 0.0;
    for (int v = 0; v < V; v++) {
        sum += *(weights_in + v);
    }
    double *weights = (double *)malloc(V * sizeof(double));
    for (int v = 0; v < V; v++) {
        *(weights + v) = *(weights_in + v) / sum * (double)V;
    }

    //Randomly init the word-topic assignments
    double **PHIS = InitPHIS(docs, Ns, M, K, weights);

    //Initialize our counts based on PHI
    double *Nmk = InitNmk(PHIS, Ns, alpha, M, K);
    double *Nk = InitNk(Nmk, M, K);

    int iter = 0;
    double diff = DBL_MAX;
    while (iter < max_iters && diff > thresh) {
        iter += 1;
        diff = DoCollapsedStep(docs, BETA, Nmk, Nk, PHIS, Ns, M, K, V, weights, false);
    }

    if (iter == max_iters) {
        printf("WARN: Convergence Failure in CVBZero -- Reached max_iters");
    }

    //Normalize Nmk so it becomes GAMMA
    double col_sum;
    for (int m = 0; m < M; m++) {
        //Make col sums
        col_sum = 0.0;
        for (int k = 0; k < K; k++) {
            col_sum += *(Nmk + m*K + k);
        }

        //Normalize
        for (int k = 0; k < K; k++) {
            *(Nmk + m*K + k) /= col_sum;
        }
    }

    //Free memory
    free(Nk);
    for (int m = 0; m < M; m++) {
        free(*(PHIS + m));
    }
    free(PHIS);
    free(docs);
    free(Ns);

    return Nmk;

}
