#ifndef _IMAGEEDITINGUTILS_H
#define _IMAGEEDITINGUTILS_H

#include <math.h>

#include <gmm.h>

namespace ImageEditingUtils {

class IImage {
public:
	virtual int getRGB(int x,int y) = 0;
	virtual void setRGB(int x,int y,int rgb) = 0;
};

const int BLACK = 0;

typedef gmm::row_matrix< gmm::rsvector<double> > SparseMatrix;
typedef std::vector<double > Vector;

int getColorInPixel(IImage& image, int y, int x, int color);

inline int getSingleColor(int rgb, int color) { return ((rgb >> (color * 8)) & 0x000000ff); }
inline int getUpper(int pixel, int n) {	return ((pixel - n) >= 0) ? (pixel - n) : -1; }
inline int getLower(int pixel, int n, int mn) { return ((pixel + n) < mn) ? (pixel + n) : -1; }
inline int getLeft(int pixel, int n) { return ((pixel % n) != 0) ? (pixel - 1) : -1; }
inline int getRight(int pixel, int n) {	return ((((pixel + 1) % n) != 0) ? (pixel + 1) : -1); }

void matrixCreate(SparseMatrix& outMatrix, int n, int mn, IImage& maskImage);
int solveLinear(const SparseMatrix& M, Vector& X, const Vector& B);
int getUpdatedRGBValue(int currentValue, int updateValue, int color);
void solveAndPaintOutput(int x0, int y0, int n, int mn,
        Vector rgbVector[3], const SparseMatrix& matrix, Vector solutionVectors[3],
		IImage& outputImage);

}

#endif
