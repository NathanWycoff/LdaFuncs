#include <Rcpp.h>
#include <iostream>
using namespace Rcpp;

IntegerMatrix CreateNwk(List docs, List Z, int V, int K) {
    int M = docs.size();
    
    // Initialize a matrix of zeros
    IntegerMatrix Nwk(V, K);
    
    //Some vars for each iter
    int N_i;
    IntegerVector doc;
    IntegerVector Z_m;
    
    //Some vars for inner loop
    int w, z;
    
    //Loop over docs and store info about word - topic relationships
    for (int m = 0; m < M; m++) {
        doc = docs(m);
        Z_m = Z(m);
        N_i = Z_m.size();
        for (int n = 0; n < N_i; n++) {
            w = doc(n);
            z = Z_m(n);
            Nwk(w - 1, z - 1) += 1;
        }
    }

    return Nwk;
}

IntegerMatrix CreateNmk(List Z, int K) {
    int M = Z.size();

    //Initialize a matrix of zeros
    IntegerMatrix Nmk(M, K);

    //Some vars for each iters
    int N_i, k;
    IntegerVector Z_m;
    
    for (int m = 0; m < M; m++) {
        Z_m = Z(m);
        N_i = Z_m.size();
        for (int n = 0; n < N_i; n++) {
            k = Z_m(n);
            Nmk(m, k-1) += 1;
        }
    }

    return Nmk;
}

// Generate a random integer between low and high, inclusive.
// Does this quickly, but does not guarantee uniform probability within the range
int FastRandInt(int low, int high) {
    int range = high - low + 1;
    int randint = rand() % range + low;
    return(randint);
}

// Initializes the list of word to topics assignemnts
List InitZ(int M, IntegerVector Ns, int K) {
    //Initialize Z, the topic assignments, to some random val.
    List Z(M);
    for (int m = 0; m < M; m++) {
        IntegerVector Z_m(Ns(m));
        for (int n = 0; n < Ns(m); n++) {
            Z_m(n) = FastRandInt(1, K);
        }
        Z(m) = Z_m;
    }
    return Z;
}

// Generate a random uniform integer.
double RandUnif() {
    double ret=((double)rand()/(double)RAND_MAX);
    return ret;
}

// Generate a random integer with the probabilites defined in rho
int RandCat(NumericVector rho) {
    //normalize rho to sum to 1
    double rho_sum = 0;
    for (int i = 0; i < rho.size(); i++) {
        rho_sum += rho(i);
    }
    for (int i = 0; i < rho.size(); i++) {
        rho(i) /= rho_sum;
    }
    
    //Iterate through rho, cumsumming until we get above the runiform value.
    double r = RandUnif();
    double cumsum = 0;
    for (int i = 0; i < rho.size(); i++) {
        cumsum += rho(i);
        if (cumsum > r) {
            return i + 1;
        }
    }
}

//Does collapsed gibbs sampling for vanilla LDA.
// mode = 0 is training, and the function will return the BETA matrix.
// mode = 1 is prediction, and the function will return the THETA matrix.
// docs_in should be an R list of numeric vectors indicating unique words in the vocab
// K = # of topics, V = size of vocab, eta is the prior on words, alpha on topics, col_iters is the number of collapsed iterations (essentially burn in), full_iters is the iters where BETA or THETA is also recorded.

// [[Rcpp::export]]
NumericMatrix TrainCollapsedGibbs(SEXP docs_in, int K, int V, SEXP eta_in, SEXP alpha_in, int col_iters, int full_iters) {
    //Convert the input S expressions to Cpp objects
    List docs = Rcpp::as<List>(docs_in);
    NumericVector eta = Rcpp::as<NumericVector>(eta_in);
    NumericVector alpha = Rcpp::as<NumericVector>(alpha_in);
    
    srand((unsigned)time(NULL));
    
    int M = docs.size();

    //Get sums of the hyperparams
    double alpha_sum = 0;
    for (int k = 0; k < K; k++) {
        alpha_sum += alpha(k);
    }
    double eta_sum = 0;
    for (int v = 0; v < K; v++) {
        eta_sum += eta(v);
    }
    
    //Get the length of each doc
    IntegerVector doc;
    IntegerVector Ns(M);
    for (int m = 0; m < M; m++) {
        doc = docs(m);
        Ns(m) = doc.size();
    }

    //Initialize Z, the word-topic assignments, and the counts
    List Z = InitZ(M, Ns, K);
    IntegerMatrix Nwk = CreateNwk(docs, Z, V, K);
    IntegerMatrix Nmk = CreateNmk(Z, K);

    //Do the actual sampling
    int iters = col_iters + full_iters;

    //Some vars for the sampling
    IntegerVector Z_m;
    int w, z, new_val;
    double a_part, b_part, normalizer;
    NumericVector rho(K);

    //Calculate some denominators
    NumericVector a_denoms(M);
    for (int i = 0; i < M; i++) {
        a_denoms(i) = Ns(i) + alpha_sum;
    }
    
    NumericVector b_denoms(K);
    for (int v = 0; v < V; v++) {
        for (int k = 0; k < K; k++) {
            b_denoms(k) += Nwk(v, k) + eta(v);
        }
    }

    //Init the BETA matrix
    NumericMatrix BETA(K,V);

    for (int iter = 0; iter < iters; iter++) {
        //Update the Z list
        for (int m = 0; m < M; m++) {
            doc = docs(m);
            Z_m = Z(m);
            for (int n = 0; n < Z_m.size(); n++) {
                w = doc(n) - 1;
                z = Z_m(n) - 1;


                //Calculate the probability of topic assignments for this word
                for (int k = 0; k < K; k++) {
                    //Another denom!
                    if (k == z) {
                        a_part = (Nmk(m, k) - 1 + alpha(k)) / a_denoms(m);
                        b_part = (Nwk(w, k) - 1 + eta(w)) / (b_denoms(k) - 1);
                    } else {
                        a_part = (Nmk(m, k) + alpha(k)) / a_denoms(m);
                        b_part = (Nwk(w, k) + eta(w)) / b_denoms(k);
                    }
                    rho(k) = a_part * b_part;
                }

                //Draw a new topic assignment
                
                new_val = RandCat(rho);
                Z_m(n) = new_val;


                //Update the counts
                //Decrement the old ones
                Nwk(w, z) -= 1;
                Nmk(m, z) -= 1;
                b_denoms(z) -= 1;

                //Increment the new ones
                Nwk(w, Z_m(n) - 1) += 1;
                Nmk(m, Z_m(n) - 1) += 1;
                b_denoms(Z_m(n) - 1) += 1;
            }

        }

        //Get a BETA draw if we're in those iters
        if (iter >= col_iters) {
            for (int k = 0; k < K; k++) {
                //Compute the normalizer for the row
                normalizer = 0;
                for (int v = 0; v < V; v ++) {
                    normalizer += Nwk(v, k) + eta(v);
                }
                //Fill the cells in this
                for (int w = 0; w < V; w++) {
                    BETA(k, w) += 1.0/double(full_iters) * (Nwk(w, k) + eta(w)) / normalizer;
                }
            }
        }
    }
    return BETA;
}

//Predict
// [[Rcpp::export]]
NumericMatrix PredictCollapsedGibbs(SEXP docs_in, NumericMatrix BETA, SEXP alpha_in, int col_iters, int full_iters) {
    //Convert the input S expressions to Cpp objects
    List docs = Rcpp::as<List>(docs_in);
    NumericVector alpha = Rcpp::as<NumericVector>(alpha_in);
    
    srand((unsigned)time(NULL));
    
    int M = docs.size();
    int K = BETA.nrow();

    //Get sums of the hyperparams
    double alpha_sum = 0;
    for (int k = 0; k < K; k++) {
        alpha_sum += alpha(k);
    }
    
    //Get the length of each doc
    IntegerVector doc;
    IntegerVector Ns(M);
    for (int m = 0; m < M; m++) {
        doc = docs(m);
        Ns(m) = doc.size();
    }

    //Initialize Z, the word-topic assignments, and the counts
    List Z = InitZ(M, Ns, K);
    IntegerMatrix Nmk = CreateNmk(Z, K);
    
    //Do the actual sampling
    int iters = col_iters + full_iters;

    //Some vars for the sampling
    IntegerVector Z_m;
    int w, z, new_val;
    double a_part, normalizer;
    NumericVector rho(K);

    //Calculate some denominators
    NumericVector a_denoms(M);
    for (int i = 0; i < M; i++) {
        a_denoms(i) = Ns(i) + alpha_sum;
    }
    
    //Init the THETA matrix
    NumericMatrix THETA(M, K);

    for (int iter = 0; iter < iters; iter++) {
        //Update the Z list
        for (int m = 0; m < M; m++) {
            doc = docs(m);
            Z_m = Z(m);
            for (int n = 0; n < Z_m.size(); n++) {
                w = doc(n) - 1;
                z = Z_m(n) - 1;


                //Calculate the probability of topic assignments for this word
                for (int k = 0; k < K; k++) {
                    //Another denom!
                    if (k == z) {
                        a_part = (Nmk(m, k) - 1 + alpha(k)) / a_denoms(m);
                    } else {
                        a_part = (Nmk(m, k) + alpha(k)) / a_denoms(m);
                    }
                    rho(k) = a_part * BETA(k, w);
                }

                //Draw a new topic assignment
                new_val = RandCat(rho);
                Z_m(n) = new_val;

                //Update the counts
                //Decrement the old ones
                Nmk(m, z) -= 1;

                //Increment the new ones
                Nmk(m, Z_m(n) - 1) += 1;
            }

        }

        //Get a THETA draw if we're in those iters
        if (iter >= col_iters) {
            for (int m = 0; m < M; m++) {
                //Compute the normalizer for the row
                normalizer = 0;
                for (int k = 0; k < K; k++) {
                    normalizer += Nmk(m, k) + alpha(k);
                }
                //Fill the cells in this
                for (int k = 0; k < K; k++) {
                    THETA(m, k) += 1.0/double(full_iters) * (Nmk(m, k) + alpha(k)) / normalizer;
                }
            }
        }
    }
    return THETA;
}
