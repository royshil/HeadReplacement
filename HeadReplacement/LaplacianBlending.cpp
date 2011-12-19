/*
 *  LaplacianBlending.cpp
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/10/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "LaplacianBlending.h"

Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
	LaplacianBlending lb(l,r,m,4);
	return lb.blend();
}

#ifdef LAPLACIAN_BLEND_MAIN

int main() {

	Mat _orange = imread("orange_fruit.jpg");
	Mat_<Vec3f> orange; _orange.convertTo(orange, orange.type(), 1.0/255.0);
	Mat _apple = imread("apple_fruit.jpg");
	Mat_<Vec3f> apple; _apple.convertTo(apple, apple.type(), 1.0/255.0);
	Mat _mask = imread("fruits_mask.png",0);
	Mat_<float> maskf; _mask.convertTo(maskf, maskf.type(), 1.0/255.0);
	
	imshow("left", orange);waitKey(1);
	imshow("right", apple);
	imshow("mask",maskf);
	imshow("Result",LaplacianBlend(orange, apple, maskf));
	waitKey(0);
}

#endif