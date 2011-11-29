#ifdef ROBUST_MATTING

#include "solver.h"


#include <stdio.h>

#include <mkl_solver.h>

//#include <mkl_spblas.h>

#include <mkl_blas.h>

#include <mkl_rci.h>

#include <stdlib.h>

#include <math.h>
 
extern "C" {

void mkl_dcsrsymv(char *uplo, int *m, double *a, int *ia, int *ja, double *x, double *y);

void DCG_INIT(int *n, double *x,double *b, int *rci_request, int *ipar, double *dpar, double *tmp);

void DCG_CHECK(int *n, double *x,double *b, int *rci_request, int *ipar, double *dpar, double *tmp);

void DCG(int *n, double *x,double *b, int *rci_request, int *ipar, double *dpar, double *tmp);

void DCG_GET(int *n, double *x, double *b, int *rci_request, int *ipar, double *dpar, double *tmp, int *itercount);

void mkl_dcsrsv(char *transa, int *m, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *y);

//void daxpy(int *n,double *alpha,double *x,int *incx,double *y,int *incy);
//
//double dnrm2(int *n,double *x,int *incx);

FILE _iob[3] = {__iob_func()[0], __iob_func()[1], __iob_func()[2]}; 

}

double solve_by_conjugate_gradient(double *A_value, int *A_rowIndex, int *A_columns, double *b, double *solution, int nonzero, int length, int iter)
{
	int RCI_request, itercount;

	/*---------------------------------------------------------------------------*/

	/* Allocate storage for the solver handle and the initial solution vector */

	/*---------------------------------------------------------------------------*/

	int ipar[128];
	double dpar[128];	
	double euclidean_norm;
	double *tmp = new double[4*length];
	double *temp = new double[length];

	/*---------------------------------------------------------------------------*/
	
	/* Initialize the solver */

	/*---------------------------------------------------------------------------*/

	DCG_INIT(&length, solution, b, &RCI_request,ipar,dpar,tmp);
	if (RCI_request!=0) goto FAILED;

	/*---------------------------------------------------------------------------*/

	/* Set the desired parameters: */

	/* INTEGER parameters: */

	/* set the maximal number of iterations to 100 */

	/* LOGICAL parameters: */

	/* run the Preconditioned version of RCI (P)CG with preconditioner C_inverse */

	/* DOUBLE PRECISION parameters */

	/* - */

	/*---------------------------------------------------------------------------*/

	ipar[4]= iter;	//number of iterations

	ipar[10]=1;

	/*---------------------------------------------------------------------------*/

	/* Check the correctness and consistency of the newly set parameters */

	/*---------------------------------------------------------------------------*/

	DCG_CHECK(&length, solution, b, &RCI_request, ipar, dpar, tmp);

	if (RCI_request!=0) goto FAILED;

	/*---------------------------------------------------------------------------*/

	/* Compute the solution by RCI (P)CG solver */

	/* Reverse Communications starts here */

	/*---------------------------------------------------------------------------*/

	RCI: DCG(&length, solution, b, &RCI_request, ipar, dpar, tmp);

	/*---------------------------------------------------------------------------*/

	/* If RCI_request=0, then the solution was found according to the requested */

	/* stopping tests. In this case, this means that it was found after 100 */

	/* iterations. */

	/*---------------------------------------------------------------------------*/

	if (RCI_request==0) goto STOP_RCI;


	/*---------------------------------------------------------------------------*/

	/* If RCI_request=1, then compute the vector A*TMP[0] */

	/* and put the result in vector TMP[N] */

	/*---------------------------------------------------------------------------*/

	if (RCI_request==1)
	{

		char U='U'; //means "upper triangle of matrix A"
		mkl_dcsrsymv(&U, &length, A_value, A_rowIndex, A_columns, tmp, &tmp[length]);

		goto RCI;

	}


	/*---------------------------------------------------------------------------*/

	/* If RCI_request=2, then do the user-defined stopping test: compute the */

	/* Euclidean norm of the actual residual using MKL routines and check if */

	/* it is less than 1.0e-8 */

	/*---------------------------------------------------------------------------*/

	if (RCI_request==2)

	{
		char U='U'; //means "upper triangle of matrix A"
		mkl_dcsrsymv(&U, &length, A_value, A_rowIndex, A_columns, solution, temp);
		
		int ione = 1;
		double mdone = -1.0e0;
		daxpy(&length, &mdone, b, &ione, temp, &ione);

		euclidean_norm = dnrm2(&length,temp,&ione);


		/*---------------------------------------------------------------------------*/

		/* The solution has not been found yet according to the user-defined stopping*/ 

		/* test. Continue RCI (P)CG iterations. */

		/*---------------------------------------------------------------------------*/

		if (euclidean_norm>1.0e-8) goto RCI;

		/*---------------------------------------------------------------------------*/

		/* The solution has been found according to the user-defined stopping test */

		/*---------------------------------------------------------------------------*/

		else goto STOP_RCI;

	}


	/*---------------------------------------------------------------------------*/

	/* If RCI_request=3, then compute apply the preconditioner matrix C_inverse */

	/* on vector TMP[2*N] and put the result in vector TMP[3*N] */

	/*---------------------------------------------------------------------------*/

	if (RCI_request==3)

	{
		double one=1.0e0;
		char matdes[3];
		char NT = 'N';	//'N': y = alpha*inv(A)*x, 'T': y = alpha*inv(A')*x
		matdes[0]='D';	// Diagonal
		matdes[1]='L';  // Lower
		matdes[2]='N';  // non-unit

		mkl_dcsrsv(&NT, &length, &one, matdes, A_value, A_rowIndex, A_columns, &A_columns[2], &tmp[2*length], &tmp[3*length]);

		goto RCI;

	}

	else

	{

	/*---------------------------------------------------------------------------*/

	/* If RCI_request=anything else, then dcg subroutine failed */

	/* to compute the solution vector: solution[N] */

	/*---------------------------------------------------------------------------*/

	goto FAILED;

	}


	/*---------------------------------------------------------------------------*/

	/* Reverse Communication ends here */

	/* Get the current iteration number */

	/*---------------------------------------------------------------------------*/

STOP_RCI:
	DCG_GET(&length, solution, b, &RCI_request, ipar, dpar, tmp, &itercount);


	/*---------------------------------------------------------------------------*/

	/* Print solution vector: solution[N] and number of iterations: itercount */

	/*---------------------------------------------------------------------------*/

	
	//for(i=0;i<length;i++) printf("solution[%d]=%6.3f\n", i, solution[i]);


	//printf("\nNumber of iterations: %d",itercount);

	goto JUMP;

FAILED:
	printf(" Solver returned error code %d\n", RCI_request);

	return -1;

JUMP:
	delete [] tmp;
	delete [] temp;
	return 0;
}

#endif