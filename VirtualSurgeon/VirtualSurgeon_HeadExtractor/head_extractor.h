#ifndef _HEAD_EXTRACTOR_H
#define _HEAD_EXTRACTOR_H

#pragma once

class GCoptimizationGridGraph;

namespace VirtualSurgeon {

class HeadExtractor {
public:
	Mat ExtractHead(Mat& im, Rect r = Rect(), Mat* skinMaskOut = NULL);
	void CreateEllipses(Mat& im, Mat &maskFace, Mat &hairMask, Point& hairEllipse);
	HeadExtractor(VirtualSurgeonParams& _p):params(_p) {};
//void FaceDotComDetection(VIRTUAL_SURGEON_PARAMS& params, Mat& im);
	void MakeMaskFromCurve(Mat_<Point2f>& curve, Mat& maskFace, Mat& hairMask, Mat& backMask, Mat& backMask1) const;
	const VirtualSurgeonParams& getParams() const { return params; } 
	void calcSegmentsLikelihood(Mat& labled_im, 
								const vector<Mat>& masks, 
								int bins,
								GCoptimizationGridGraph* gc,
								InputArray vert_edge_score,
								InputArray horiz_edge_score, 
								int* score_matrix, 
								Mat_<char>& hard_constraints);
	
	
private:
	void calcHistogramWithMask(vector<MatND>& hist, Mat &im, vector<Mat>& mask, float _max, int win_size = 10, int histCompareMethod = CV_COMP_CORREL, vector<Mat> backProj = vector<Mat>(), vector<Mat> hists = vector<Mat>());
	void create2DGaussian(Mat& im, double sigma_x, double sigma_y, Point mean);
	Mat gabor_fn(double sigma, int n_stds, double theta, double freq, double phase, double gamma);
	void getSobels(Mat& gray, Mat& grayInt, Mat& grayInt1);
	int head_extract_main(int argc, char** argv);
	void make_gabor_bank(vector<Mat>& filter_bank, int bank_size, double sigma, int n_stds, double freq, double phase, double gamma);
	void NaiveRelabeling(Size s, vector<Mat>& backP, vector<Mat>& maskA);
	//void takeBiggestCC(Mat& mask, Mat& bias = Mat());

	template<typename T>
	void getEdgesForGC(Mat& gray, Mat_<T>& horiz, Mat_<T>& vert);
	template<typename T>
	void getEdgesUsingTextons(Mat& lables, Mat& descriptors, Mat_<T>& horiz, Mat_<T>& vert);
	
	void CurveUserAidOpenCV();
	
	VirtualSurgeonParams& params;
};

}//ns

#endif