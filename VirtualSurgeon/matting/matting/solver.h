#ifdef ROBUST_MATTING

#ifndef SOLVER_H
#define SOLVER_H

//solve Ax = b by conjugate gradient (CG),
//where A is a symmetric matrix

//Note:
//(1)Remember to initialize solution
//(2)Diagonal items should be explicitly declared (even if it is zero)
//(3)vector index starts form 1 instead of 0
double solve_by_conjugate_gradient(double *A_value, int *A_rowIndex, int *A_columns, double *b, double *solution, int nonzero, int length, int iter = 100);

/*
A = [ 7,  0,  1,  0,  0,  2,  7,  0
      0, -4,  8,  0,  2,  0,  0,  0
	  1,  8,  1,  0,  0,  0,  0,  5
	  0,  0,  0,  7,  0,  0,  9,  0
	  0,  2,  0,  0,  5,  1,  5,  0
	  2,  0,  0,  0,  1, -1,  0,  5
	  7,  0,  0,  9,  5,  0, 11,  0
	  0,  0,  5,  0,  0,  5,  0,  5]
example:
	double rhs[8] = {15, 10, 2, 9, 10, 3, 23, 5};
	int ia[9]={1,5,8,10,12,15,17,18,19};
	int ja[18]={1, 3, 6,7, 
				2,3, 5,
				3, 8,
				4, 7,
				5,6,7,
				6, 8,
				7,
				8};

	double a[18]={ 7.0e0, 1.0e0, 2.0e0, 7.0e0,
					-4.0e0, 8.0e0, 2.0e0,
					1.0e0, 5.0e0,	
					7.0e0, 9.0e0,
					5.0e0, 1.0e0, 5.0e0,
					-1.0e0, 5.0e0,
					11.0e0,
					5.0e0};
	double solution[8]={};
	solve_by_conjugate_gradient(a, ia, ja, rhs, solution, 18, 8);
*/

#endif


#endif