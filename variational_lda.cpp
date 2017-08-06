#include <Rcpp.h>
#include <iostream>
#include <math.h>
#include <float.h>

#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029L
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024L
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766L
#endif

//TODO: Don't reinit PHI and GAMMA on each iter; see how it changes things.

using namespace Rcpp;

/** The digamma function in long double precision.
* @param x the real value of the argument
* @return the value of the digamma (psi) function at that point
* @author Richard J. Mathar
* @since 2005-11-24
*/
long double digammal(long double x)
{
	/* force into the interval 1..3 */
	if( x < 0.0L )
		return digammal(1.0L-x)+M_PIl/tanl(M_PIl*(1.0L-x)) ;	/* reflection formula */
	else if( x < 1.0L )
		return digammal(1.0L+x)-1.0L/x ;
	else if ( x == 1.0L)
		return -M_GAMMAl ;
	else if ( x == 2.0L)
		return 1.0L-M_GAMMAl ;
	else if ( x == 3.0L)
		return 1.5L-M_GAMMAl ;
	else if ( x > 3.0L)
		/* duplication formula */
		return 0.5L*(digammal(x/2.0L)+digammal((x+1.0L)/2.0L))+M_LN2l ;
	else
	{
		/* Just for your information, the following lines contain
		* the Maple source code to re-generate the table that is
		* eventually becoming the Kncoe[] array below
		* interface(prettyprint=0) :
		* Digits := 63 :
		* r := 0 :
		* 
		* for l from 1 to 60 do
		* 	d := binomial(-1/2,l) :
		* 	r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
		* 	evalf(r) ;
		* 	print(%,evalf(1+Psi(1)-r)) ;
		*o d :
		* 
		* for N from 1 to 28 do
		* 	r := 0 :
		* 	n := N-1 :
		*
 		*	for l from iquo(n+3,2) to 70 do
		*		d := 0 :
 		*		for s from 0 to n+1 do
 		*		 d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
 		*		od :
 		*		if 2*l-n > 1 then
 		*		r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
 		*		fi :
 		*	od :
 		*	print(evalf((-1)^n*2*r)) ;
 		*od :
 		*quit :
		*/
		static long double Kncoe[] = { .30459198558715155634315638246624251L,
		.72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
		.27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
		.17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
		.11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
		.83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
		.59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
		.42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
		.304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
		.21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
		.15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
		.11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
		.80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
		.58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
		.41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;

		register long double Tn_1 = 1.0L ;	/* T_{n-1}(x), started at n=1 */
		register long double Tn = x-2.0L ;	/* T_{n}(x) , started at n=1 */
		register long double resul = Kncoe[0] + Kncoe[1]*Tn ;

		x -= 2.0L ;

		for(int n = 2 ; n < sizeof(Kncoe)/sizeof(long double) ;n++)
		{
			const long double Tn1 = 2.0L * x * Tn - Tn_1 ;	/* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
			resul += Kncoe[n]*Tn1 ;
			Tn_1 = Tn ;
			Tn = Tn1 ;
		}
		return resul ;
	}
}


// Generate a random uniform integer.
double RandUnif() {
    double ret=((double)rand()/(double)RAND_MAX);
    return ret;
}

//Initialize PHI, the variational parameter for word topic assignment for one document. Note that PHI is assigned to before it is read, so no need to init to any particular value.
double *InitPHI(int N, int K) {
    double *PHI = (double *)malloc(N * K * sizeof(double));
    return PHI;
}

//Initilialize GAMMA, the variational parameter for document topic mixtures.
double *InitGAMMA(int M, int K, int *Ns, double *alpha) {
    double *GAMMA = (double *)malloc(M * K * sizeof(double));
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            *(GAMMA + m * K + k) = (double)*(Ns + m)/(double)K + *(alpha + k);
        }
    }
    return GAMMA;
}

//Initilize Storage for BETA, the document topic matrix, and fill it with uniform random numbers then normalize.
double *InitBETA(int K, int V) {
    double *BETA = (double *)malloc(K * V * sizeof(double));
    double *row_sums = (double *)malloc(K * sizeof(double));
    double new_val;

    //Generate the random stuff
    for (int k = 0; k < K; k++) {
        *(row_sums + k) = 0;
        for (int v = 0; v < V; v++) {
            new_val = RandUnif();
            *(BETA + k*V + v) = new_val;
            *(row_sums + k) += new_val;
        }
    }
    
    //Normalize the rows to sum to 1
    for (int k = 0; k < K; k++) {
        for (int v = 0; v < V; v++) {
            *(BETA + k*V + v) /= *(row_sums + k);
        }
    }
    return BETA;
}

//Get l_{infty} norm (max absolute difference) between two vectors of length N
double LinfDistance(double *vec1, double *vec2, int N) {
    double max = 0;
    double absdiff;
    double val1, val2;
    for (int n = 0; n < N; n++) {
        val1 = *(vec1 + n);
        val2 = *(vec2 + n);
        absdiff = std::abs(val1 - val2);
        if (absdiff > max) {
            max = absdiff;
        }
    }
    return max;
}

//Conduct the variational E step for one document, updating PHI and gamma inplace
//
// PHI is the matrix for a certain doc, gamma is the appropriate row of the GAMMA matrix, and beta is the vector of probabilities associated with each word corresponding to the rows of PHI, (or actually, something propto that. NO LONGER SURE ABT THIS)
// We check convergence using the inf norm of gamma
// K is the topic count, N is the word count
void DoEStep(double *PHI, double *gamma, double * BETA, int * doc, double * alpha, int K, int V, int N, double thresh, int max_iter) {
    double diff = DBL_MAX;
    int iter = 0;
    
    //Init some memory
    double *gamma_old = (double *)malloc(K * N * sizeof(double));
    double *col_sums = (double *)malloc(K * sizeof(double));
    
    while (diff > thresh && iter < max_iter) {
        // Store the old gamma value to check convergence
        for (int k = 0; k < K; k++) {
            *(gamma_old + k) = *(gamma + k);
        }
     
        //update PHI
        for (int k = 0; k < K; k ++) {
            *(col_sums + k) = 0;
        }
        double row_sum;
        double *update = (double *)malloc(K * sizeof(double));
        for (int n = 0; n < N; n++) {
            //Reset some things
            row_sum = 0.0;
            //Calculate up to a constant
            for (int k = 0; k < K; k++) {
                //We need to subtract 1 because our documents are 1 indexed.
                *(update + k) = *(BETA + k*V + *(doc + n) - 1) * exp(digammal(*(gamma + k)));
                *(PHI + n*K + k) = *(update + k);
                row_sum += *(update+k);
            }
            //Now normalize it
            for (int k = 0; k < K; k++) {
                *(PHI + n*K + k) /= row_sum;
                *(col_sums + k) += *(update + k) / row_sum;
            }
        }
        
        //Update gamma
        for (int k = 0; k < K; k++) {
            *(gamma + k) = *(alpha + k) + *(col_sums + k);
        }


        //Check convergence
        diff = LinfDistance(gamma_old, gamma, K);
    }
}

//M Step - BETA only
double *DoMStep(double *BETA, double **PHIS, int *Ns, int **docs, int K, int V, int M, double lambda) {
    double *row_sums = (double *)malloc(K * sizeof(double));
    //Reset BETA to be the prior on words, lambda;
    for (int k = 0; k < K; k++) {
        *(row_sums + k) = V * lambda;
        for (int v = 0; v < V; v++) {
            *(BETA + k*V + v) = lambda;
        }
    }
    
    //Increment by the new vals
    double *PHI;
    int *doc;
    int N, w;
    for (int m = 0; m < M; m++) {
        //Store some vals for this doc
        PHI = *(PHIS + m);
        doc = *(docs + m);
        N = *(Ns + m);
        
        //Update BETA based on the new PHI
        for (int n = 0; n < N; n++) {
            w = *(doc + n) - 1;
            for (int k = 0; k < K; k++) {
                *(BETA + k*V + w) += *(PHI + n * K + k);
                *(row_sums + k) += *(PHI + n * K + k);
            }
        }
    }

    //Normalize the rows to sum to 1
    for (int k = 0; k < K; k++) {
        for (int v = 0; v < V; v++) {
            *(BETA + k*V + v) /= *(row_sums + k);
        }
    }
    return BETA;
}

/*
 * A variational EM algorithm for fitting a vanilla LDA model.
 */
double *VariationalEM(int **docs, int *Ns, double *alpha, double lambda, int K, int V, int M, double e_thresh, double em_thresh, int em_max_iters, int e_max_iters, int seed) {
    //Initiliaze the differences
    double em_diff = DBL_MAX;
    int iter = 0;
    double *em_diffs = (double *)malloc(K * sizeof(double));

    //Set our seed
    srand(seed);

    //Initialize some variational params
    double *GAMMA = InitGAMMA(M, K, Ns, alpha);
    double **PHIS = (double **)malloc(M * sizeof(int *));
    for (int m = 0; m < M; m++) {
        *(PHIS + m) = InitPHI(*(Ns + m), K);
    }

    //Since we're using BETA to check convergence, we need to keep the old one in memory.
    double *BETA = InitBETA(K, V);
    double *new_BETA = (double *)malloc(K * V * sizeof(double));
    double *beta_holder;

    while (em_diff > em_thresh && iter < em_max_iters) {
        iter += 1;
        std::cout << "Iter:";
        std::cout << iter << std::endl;
        //E Step over all docs
        for (int m = 0; m < M; m++) {
            DoEStep(*(PHIS + m), GAMMA + m*K, BETA, *(docs + m), alpha, K, V, *(Ns + m), e_thresh, e_max_iters);
        }

        //M Step
        DoMStep(new_BETA, PHIS, Ns, docs, K, V, M, lambda);

        //Check convergence
        em_diff = 0;
        for (int k = 0; k < K; k++) {
            *(em_diffs + k) = LinfDistance(new_BETA + k*V, BETA + k*V, V);
            if (*(em_diffs + k) > em_diff) {
                em_diff = *(em_diffs + k);
            }
        }
        beta_holder = BETA;
        BETA = new_BETA;
        new_BETA = beta_holder;
    }

    return BETA;
}

//An R Wrapper for a C variational EM function, just convert from R-like cpp objects to pointers in memory.
// [[Rcpp::export]]
NumericMatrix RVariationalEM(List docs_in, NumericVector alpha, double lambda, int K, int V, double e_thresh, double em_thresh, int em_max_iters, int e_max_iters, int seed) {
    int M = docs_in.size();
    IntegerVector current;
    int **docs = (int **)malloc(M * sizeof(int *)); 
    int *Ns = (int *) malloc(M * sizeof(int));
    for (int m = 0; m < M; m++) {
        current = docs_in(m);
        *(docs + m) = current.begin();
        *(Ns + m) = current.size();
    }
    double *BETA = VariationalEM(docs, Ns, alpha.begin(), lambda, K, V, M, e_thresh, em_thresh, em_max_iters, e_max_iters, seed);
    return NumericMatrix(V, K, BETA);
}
