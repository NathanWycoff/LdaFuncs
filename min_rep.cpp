#include <iostream>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
void MinRep(List docs_in, int V) {
    int M = docs_in.size();
    IntegerVector current;
    int **docs = (int **)malloc(M * sizeof(int *)); 
    int *Ns = (int *) malloc(M * sizeof(int));
    for (int m = 0; m < M; m++) {
        current = docs_in(m);
        *(docs + m) = (int *)current.begin();
        *(Ns + m) = (int)current.length();
    }

    //Copy the docs to avoid the world's weirdest bug.
    //int **docs = (int **)malloc(M * sizeof(int *));
    //IntegerVector current;
    //int *Ns = (int *)malloc(M * sizeof(int));
    //int *doc;
    //for (int m = 0; m < M; m++) {
    //    current = docs_in(m);
    //    *(Ns + m) = current.size();
    //    doc = (int *)malloc(*(Ns + m) * sizeof(int));
    //    for (int n = 0; n < *(Ns + m); n++) {
    //        *(doc + n) = current(n);
    //    }
    //    *(docs + m) = doc;
    //}
    
    int N, w;
    int *doc;
    for (int m = 0; m < M; m++) {
        N = *(Ns + m);
        doc = *(docs + m);
        for (int n = 0; n < N; n++) {
            w = *(doc + n);
            if (w > V || w < 1) {
                std::cout << "Oh fuck" << std::endl;
                std::cout << "on doc ";
                std::cout << m << std::endl;
                std::cout << "So here\'s the doc:" << std::endl;
                for (int ni = 0; ni < N; ni++) {
                    std::cout << *(doc + ni);
                    std::cout << ", ";
                }
                std::cout << "And here\'s the integer array:" << std::endl;
                current = docs_in(m);
                for (int ni = 0; ni < N; ni ++) {
                    std::cout << current(ni);
                    std::cout << ", ";
                }
                return;
            }
        }
    }
    free(docs);
    free(Ns);
}
