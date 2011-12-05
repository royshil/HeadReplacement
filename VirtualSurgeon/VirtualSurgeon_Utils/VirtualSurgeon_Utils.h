#pragma once

#define _PI 3.14159265

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace VirtualSurgeon {

	class VirtualSurgeonFaceData {
	public:
		string filename;
		std::string path_to_exe; //needed for loading cascade classifiers 

		//face data
		cv::Point li,ri,center,nose,mouth_left,mouth_right;
		double yaw;
		double roll;
		double pitch;

		bool FaceDotComDetection(cv::Mat& im);
		void DetectEyes(cv::Mat& im);
	};

class VirtualSurgeonParams : public VirtualSurgeonFaceData {
public:
	//algorithm data
	double gb_sig;
	double gb_freq;
	double gb_phase;
	double gb_gamma;
	int gb_nstds;
	int gb_size;
	int km_numc;
	int com_winsize;
	double com_thresh;
	string groundtruth;
	int com_add_type;
	int com_calc_type;
	double im_scale_by;
	int gc_iter;
	int km_numt;
	bool doScore;
	int relable_type;
	bool doPositionInKM;
	bool doInitStep;
	int num_cut_backp_iters;
	bool do_alpha_matt;
	int alpha_matt_dilate_size;
	double hair_ellipse_size_mult;
	bool do_eq_hist;
	bool consider_pixel_neighbourhood;
	bool do_two_segments;
	bool do_kmeans;
	double head_mask_size_mult;
	int num_DoG;
	bool do_two_back_kernels;

	double snake_snap_weight_edge;
	double snake_snap_weight_direction;
	double snake_snap_weight_consistency;
	double snake_snap_weight_tension;
	int snake_snap_edge_len_thresh;
	int snale_snap_total_width_coeff;

	int poisson_cloning_band_size;

	bool use_hist_match_hs;
	bool use_hist_match_rgb;
	bool use_overlay;
	bool use_warp_rigid;
	bool use_warp_affine;
	bool use_double_warp;

	bool no_gui;
	int wait_time;

	bool is_female;

	std::string output_filename;

	bool blur_on_resize;
	bool two_way_recolor;

	Mat_<Point2f> m_curve;
	

	void InitializeDefault();
	void ParseParams(int argc, const char** argv);
	void PrintParams();
	void face_grab_cut(cv::Mat& orig, cv::Mat& mask, int iters, int dilate_size = 30);
	void PoissonImageEditing(const cv::Mat& back, const cv::Mat& backMask, const cv::Mat& front, const cv::Mat& frontMask, bool doLaplacian = true, bool doBounding = true);
	Mat_<Point2f>& LoadHeadCurve(Mat& im, Rect r);
	std::string GenerateUniqueID();
};


void FindBoundingRect(Rect& faceRect, const Mat& headMask);
void takeBiggestCC(Mat& , Mat bias = Mat());

}//ns