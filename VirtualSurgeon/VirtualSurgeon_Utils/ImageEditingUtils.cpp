#include "stdafx.h"
#include "ImageEditingUtils.h"

#include <iostream>
using namespace std;

namespace ImageEditingUtils {

int getColorInPixel(IImage& image, int y, int x, int color) {
	int rgb = image.getRGB(x, y);
	return getSingleColor(rgb, color);
}

void matrixCreate(SparseMatrix& outMatrix, int n, int mn, IImage& maskImage) {
	cout << "ImageEditingUtils::matrixCreate("<<((intptr_t)(&outMatrix))<<","<<n<<","<<mn<<","<<((intptr_t)(&maskImage))<<")"<<endl;
	for (int pixel = 0; pixel < mn; pixel++) {
		int pxlX = pixel % n;
		int pxlY = (int) floor((float)pixel / (float)n);
		int maskRGB = maskImage.getRGB(pxlX, pxlY);
		if ((maskRGB & 0x00ffffff) == BLACK) {
			outMatrix(pixel,pixel) = 1;
		} else {
			// add 1s in sides
			int numOfNeighbors = 0;
			int neighborsArray[4] = {0};
			neighborsArray[0] = getUpper(pixel, n);
			neighborsArray[1] = getLower(pixel, n, mn);
			neighborsArray[2] = getRight(pixel, n);
			neighborsArray[3] = getLeft(pixel, n);
			for (int j = 0; j < 4; j++) {
				if (neighborsArray[j] >= 0) {
					outMatrix(pixel,neighborsArray[j]) = 1;
					numOfNeighbors++;
				}
			}

			//add -4, -3 or -2 in middle
			outMatrix(pixel,pixel) = (-1) * numOfNeighbors;
		}
	}
}

/**
 * Solve linear equation system
 */
int solveLinear(const SparseMatrix& M, Vector& X, const Vector& B) {
//	gsl_permutation * p = gsl_permutation_alloc(4);
//	int s;
//	gsl_linalg_LU_decomp(A, p, &s);
//	gsl_linalg_LU_solve(A, p, b, X);
//	return s;

//	gmm::identity_matrix PS;
//	gmm::identity_matrix PR;

	gmm::ilut_precond<SparseMatrix > PR(M, 10, 1e-2);

	gmm::iteration iter(1E-8);  // defines an iteration object, with a max residu of 1E-8

	gmm::gmres(M, X, B, PR, 50, iter);  // execute the GMRES algorithm

//	gmm::least_squares_cg(M,X,B,iter);

//	gmm::bicgstab(M,X,B,P,iter);

//	gmm::cg(M,X,B,PS,PR,iter);

	return 1;
}

int getUpdatedRGBValue(int currentValue, int updateValue, int color) {
	int colorOffest = (color * 8);
	int colorInRGB = (updateValue << colorOffest);
	int bitmask = 0xffffffff ^ (0x000000ff << colorOffest);
	int roomForUpdated = (currentValue & bitmask);
	int updatedRGB = (roomForUpdated | colorInRGB);
	return updatedRGB;
}

void solveAndPaintOutput(int x0, int y0, int n, int mn,
        Vector rgbVector[3], const SparseMatrix& matrix, Vector solutionVectors[3],
		IImage& outputImage) {
	for (int color = 0; color < 3; color++) {
		cout << "Solving..."<<endl;
		// solve equations set for current color
		if (solveLinear(matrix, solutionVectors[color], rgbVector[color]) == 0) {
			cout << "FAIL main(): matrix.solve() failed with color: "<<color<<endl;
			return;
		} else {
			cout << "Done solving color "<<color<<endl;
		}

		// fill output image
		for (int pixel = 0; pixel < mn; pixel++) {
			int y = pixel / n;
			int x = pixel - n * y;

			int updateVal = (int)solutionVectors[color][pixel];

			if (updateVal > 255) {
				updateVal = 255;
			}
			else if (updateVal < 0) {
				updateVal = 0;
			}
			int updatedRGBVal = getUpdatedRGBValue(outputImage.getRGB(x + x0, y + y0), updateVal, color);
			outputImage.setRGB(x + x0, y + y0, updatedRGBVal);
		}
		cout<<"Done applying color "<<color<<" to output"<<endl;
	}
}


}
