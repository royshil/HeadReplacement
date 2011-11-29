#include "StdAfx.h"

#include "head_extractor.h"

#include "../tclap-1.2.0/include/tclap/CmdLine.h"
using namespace TCLAP;

#include "GCoptimization.h"

#include "matting.h"
#include <float.h>
#include <fstream>

//#include "../BeTheModel/util.h"

#define _PI 3.14159265
#define TWO_PI 6.2831853

#define RELABLE_HIST_MAX 0
#define RELABLE_GRAPHCUT 1

void StartCurveWindow(const cv::Mat& im_bgr, 
						Mat_<Point2f>& curve,
						Mat& _maskFace, 
						Mat& _hairMask, 
						Mat& _backMask, 
						Mat& _backMask1,
						const VirtualSurgeon::HeadExtractor& _he);

void StartSegmentationWindow(const Mat& im_bgr, 
							 Mat& _tmp,
							 VirtualSurgeon::HeadExtractor& _he,
							 vector<Mat>& _maskA,
							 Mat_<char>& _hard_constraints,
							 Mat& _labels,
							 int _num_lables,
							 const Mat& _bias,
							 GCoptimizationGridGraph& _gc);

namespace VirtualSurgeon {

//void checkArgF(const char* arg, const char* c, int l, double* val) {
//	if(strncmp(arg,c,l)==0) {
//		char __s[10] = {0};
//		strncpy_s(__s,10,arg+l+1,10);
//		*val = atof(__s);
//	}
//}
//
//void checkArgI(const char* arg, const char* c, int l, int* val) {
//	if(strncmp(arg,c,l)==0) {
//		char __s[10] = {0};
//		strncpy_s(__s,10,arg+l+1,10);
//		*val = atoi(__s);
//	}
//}

struct MouseEvent
{
    MouseEvent() { event = -1; buttonState = 0; }
    Point pt;
    int event;
    int buttonState;
};

static void onMouse(int event, int x, int y, int flags, void* userdata)
{
//	cout << "event " << event << ", [x,y] " << x <<","<<y<< ", flags " << flags <<endl;
    MouseEvent* data = (MouseEvent*)userdata;
    data->event = event;
    data->pt = Point(x,y);
    data->buttonState = flags;
}


/*
filter - the output filter
sigma - is the sigma of the Gaussian envelope
n_stds - number of sigmas in Gaussian
freq - represents the wavelength of the cosine factor, 
theta - represents the orientation of the normal to the parallel stripes of a Gabor function, 
phase - is the phase offset, 
gamma - is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.
*/
Mat HeadExtractor::gabor_fn(double sigma, int n_stds, double theta, double freq, double phase, double gamma) {
	double sigma_x = sigma;
	double sigma_y = sigma / gamma;

	//int sz_x=(int)floor((double)n_stds * sigma_x + 1.0);
	//int sz_y=(int)floor((double)n_stds * sigma_y + 1.0);
	//sz_x = MAX(sz_x,sz_y);
	//sz_y = MAX(sz_x,sz_y);

	//int sz_x_2 = (int)floor((double)sz_x / 2.0);
	//int sz_y_2 = (int)floor((double)sz_y / 2.0);

	double d_sz_x_2 = MAX(abs(n_stds*sigma_x*cos(theta)),abs(n_stds*sigma_y*sin(theta)));
	int sz_x_2 = (int)ceil(MAX(1.0,d_sz_x_2));
	double d_sz_y_2 = MAX(abs(n_stds*sigma_x*sin(theta)),abs(n_stds*sigma_y*cos(theta)));
	int sz_y_2 = (int)ceil(MAX(1.0,d_sz_y_2));

	int sz_x = sz_x_2 * 2 + 1;
	int sz_y = sz_y_2 * 2 + 1;

	//cerr << sz_x_2 << "->" << sz_x << "," << sz_y_2 << "->" << sz_y << endl;

	//int totSize = sz_x * sz_y;
	//double* gb_d = new double[totSize];
	//memset(gb_d,0,totSize*sizeof(double));

	Mat filter(sz_y,sz_x,CV_32FC1);
#ifdef _PI
	cerr << "Mat("<<sz_y<<","<<sz_x<<",CV_32FC1)"<<endl;
#endif

	double  x_theta;
	double  y_theta;
	double  x_theta_sq;
	double  y_theta_sq;
	double  sigma_x_sq = sigma_x * sigma_x;
	double  sigma_y_sq = sigma_y * sigma_y;

	int step = sz_x;
	int count = 0;
	float* _f_ptr = filter.ptr<float>();
	for(int y=-sz_y_2;y<=sz_y_2;y++) {
		for(int x=-sz_x_2;x<=sz_x_2;x++) {
			// Rotation 
			x_theta=x*cos(theta)+y*sin(theta);
			y_theta=-x*sin(theta)+y*cos(theta);
			x_theta_sq = x_theta * x_theta;
			y_theta_sq = y_theta * y_theta;

			//((float*)(filter.data))[(y+sz_y_2) * step + (x+sz_x_2)] = 
			_f_ptr[count++] = 
				(float)(exp(-.5*(x_theta_sq/sigma_x_sq+y_theta_sq/sigma_y_sq))*cos(TWO_PI*freq*x_theta + phase));
		}
	}

	//memcpy(filter.data,gb_d,totSize * sizeof(double));

	//delete[] gb_d;
	return filter;
}

void HeadExtractor::make_gabor_bank(vector<Mat>& filter_bank, int bank_size, double sigma, int n_stds, double freq, double phase, double gamma) {
	cerr << "make gabor bank.. ";

	//int bank_size = filter_bank.size();
	cerr << "size: "<<bank_size;
	vector<double> angs;
	for(int i=0;i<bank_size;i++) angs.push_back((double)(i) * _PI / bank_size);

	//namedWindow("tmp2");
	for(int i=0;i<bank_size;i++) {
		cerr << ".";
		filter_bank.push_back(gabor_fn(sigma,n_stds,angs[i],freq,phase,gamma));
		//imshow("tmp2",(filter_bank[i] + 1.0)/2.0);
		//waitKey();
	}

	cerr << " done"<<endl;
}

//im - the image to calc hist for
//mask - the area in the image to calc over
// range - the highest value for the bins
void HeadExtractor::calcHistogramWithMask(vector<MatND>& hist, Mat &im, vector<Mat>& mask, float _max, int win_size, int histCompareMethod, vector<Mat> backProj, vector<Mat> hists) {
    int bins = (int)_max;
    int histSize[1] = {bins};
    float range[2] = { 0, _max };
    float* ranges[] = { range };
    int channels[1] = {0};

	if(backProj.size() != hist.size()) {
		backProj = vector<Mat>(hist.size());
	}
	if(hists.size() != hist.size()) {
		hists = vector<Mat>(hist.size());
	}

	for(unsigned int i=0;i<hist.size();i++) {

		calcHist((const Mat*)( &(im) ), 1, channels, mask[i],
			hist[i], 1, histSize, (const float**)(ranges),
			true, // the histogram is uniform
			false );

		double maxVal=0;
		minMaxLoc(hist[i], 0, &maxVal, 0, 0);

#ifdef _PI
		if(!params.no_gui) {
			int scale = 10;
			Mat histImg = Mat::zeros(200, bins*scale, CV_8UC1);

			for( int s = 0; s < bins; s++ )
			{
				float binVal = hist[i].at<float>(s);
				if(binVal <= 0) continue;
				//int intensity = cvRound(binVal*255/maxValue);
				rectangle( histImg, Point(s*scale, 199),
							 Point( (s+1)*scale - 1, 200 - (int)floor((double)binVal*200.0/maxVal)),
							 Scalar::all(255),
							 CV_FILLED );
			}

			imshow("tmp",histImg);
		}
#endif

	}

	//Mat backProject;
	//calcBackProject((const Mat*)(&im),1,channels,hist,backProject,(const float**)ranges);
	//imshow("tmp2",backProject);

	Mat paddedIm(im.rows+win_size,im.cols+win_size,im.type());
	paddedIm.setTo(Scalar(0));//on the borders there will be bias...
	Mat _roi = paddedIm(Rect(win_size/2,win_size/2,im.cols,im.rows));
	im.copyTo(_roi);

	for(int i=0;i<hist.size();i++) {
		normalize(hist[i],hist[i]);

		backProj[i] = Mat(paddedIm.size(),CV_64FC1,Scalar(0));
		hists[i] = Mat(countNonZero(mask[i]),bins,CV_32FC1);
	}

	vector<int> cnt(backProj.size(),0);
	for(int y=0;y<paddedIm.rows-win_size;y++) {
		//double* _bp_ptr = backProj.ptr<double>(y);

		for(int x=0;x<paddedIm.cols-win_size;x++) {
			Mat window = paddedIm(Rect(x,y,win_size,win_size));
			MatND winhist;
			calcHist((const Mat*)( &(window) ), 1, channels, Mat(),
				winhist, 1, histSize, (const float**)(ranges),
				true, // the histogram is uniform
				false );
			normalize(winhist,winhist);

			for(int i=0;i<backProj.size();i++) {
				double score = compareHist(winhist,hist[i],histCompareMethod);
				backProj[i].at<double>(y,x) = score;
				if(mask[i].at<uchar>(y,x) > 0) {
					memcpy(hists[i].ptr<float>(cnt[i]),winhist.data,sizeof(float)*bins);
					cnt[i]++;
				}
			}
		}
	}

	for(int i=0;i<backProj.size();i++) {
		backProj[i] = backProj[i](Rect(win_size/2,win_size/2,im.cols,im.rows));
#ifdef _PI
		if(!params.no_gui) {
			double maxV,minV;
			minMaxLoc(backProj[i],&minV,&maxV);
			imshow("tmp1",(backProj[i] - minV) / (maxV - minV));

			waitKey(params.wait_time);
		}
#endif
	}
}

void HeadExtractor::getSobels(Mat& gray, Mat& grayInt, Mat& grayInt1) {
	Mat _tmp,_tmp1,gray32f;
	
	gray.convertTo(gray32f,CV_32FC1,1.0/255.0);

	GaussianBlur(gray32f,gray32f,Size(15,15),.75);

	Sobel(gray32f,_tmp,-1,2,0,3);	//sobel for dx
	Sobel(gray32f,_tmp1,-1,2,0,3,-1.0);	//sobel for -dx
	//Canny(gray,_tmp,50.0,150.0);
	_tmp = abs(_tmp) + abs(_tmp1);

	if(!params.no_gui) {
		imshow("tmp1",_tmp1);
		imshow("tmp",_tmp); 
		waitKey(params.wait_time);
	}

	double maxVal,minVal;
	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log((_tmp - minVal) / (maxVal - minVal),_tmp);
	_tmp = -_tmp * 0.17;

	_tmp.convertTo(grayInt1,CV_32SC1);
	
	//grayInt = grayInt * 5;

	//filter2D(gray,_tmp,CV_32F,gk.t());
	Sobel(gray32f,_tmp,-1,0,2,3);	//sobel for dy
	Sobel(gray32f,_tmp1,-1,0,2,3,-1.0);	//sobel for -dy
	//Canny(gray,_tmp,50.0,150.0);
	_tmp = abs(_tmp) + abs(_tmp1);

	if(!params.no_gui) {
		imshow("tmp",_tmp); waitKey(params.wait_time);
	}

	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log((_tmp - minVal) / (maxVal - minVal),_tmp);
	_tmp = -_tmp * 0.17;
	_tmp.convertTo(grayInt,CV_32SC1);

}

	template<typename T>
void HeadExtractor::getEdgesForGC(Mat& gray, Mat_<T>& horiz, Mat_<T>& vert) {
	double sigma = 2.5; 
	double two_sigma_sq = 2*sigma*sigma;

	horiz.create(gray.size());
	vert.create(gray.size());
				
	double lambda = 1.0f;

	 //exp function went bad...
	Mat gray_x_p_1 = gray(Range(0,gray.rows),Range(1,gray.cols));
	Mat gray_x_m_1 = gray(Range(0,gray.rows),Range(0,gray.cols-1));
	Mat gray_x_d = gray_x_p_1 - gray_x_m_1; //drivative on x
	Mat_<T> gray_x_d_64f; gray_x_d.convertTo(gray_x_d_64f,gray_x_d.type());
	Mat _a = Mat(gray_x_d_64f.mul(gray_x_d_64f) / -two_sigma_sq);
	Mat _b = horiz(Range(0,horiz.rows),Range(0,horiz.cols-1));
	exp(_a, _b);

	Mat gray_y_p_1 = gray(Range(1,gray.rows),Range(0,gray.cols));
	Mat gray_y_m_1 = gray(Range(0,gray.rows-1),Range(0,gray.cols));
	Mat gray_y_d = gray_y_p_1 - gray_y_m_1; //derivative on y
	Mat_<T> gray_y_d_64f; gray_y_d.convertTo(gray_y_d_64f,gray_y_d.type());
	_a = gray_y_d_64f.mul(gray_y_d_64f) / -two_sigma_sq;
	_b = vert(Range(0,gray.rows-1),Range(0,gray.cols));
	exp(_a, _b);
	

	/*

	uchar* gray_ptr_p1;
	double* vert_ptr_p1;
	uchar* gray_ptr_m1;
	double* vert_ptr_m1;
	for(int y=0;y<gray.rows;y++) {
		uchar* gray_ptr = gray.ptr<uchar>(y);
		double* horiz_ptr = horiz.ptr<double>(y);
		double* vert_ptr = vert.ptr<double>(y);

		if(y<gray.rows-1) {
			gray_ptr_p1 = gray.ptr<uchar>(y+1);
			vert_ptr_p1 = vert.ptr<double>(y+1);
		}
		if(y>0) {
			gray_ptr_m1 = gray.ptr<uchar>(y-1);
			vert_ptr_m1 = vert.ptr<double>(y-1);
		}


		for(int x=0;x<gray.cols;x++) {
			if(x>0) {
				double d = (double)gray_ptr[x] - (double)gray_ptr[x-1];
				horiz_ptr[x-1] += lambda*(exp(-(d*d)/two_sigma_sq));
			}
			if(y>0) {
				double d = (double)gray_ptr[x] - (double)gray_ptr_m1[x];
				vert_ptr_m1[x] += lambda*(exp(-(d*d)/two_sigma_sq));
			}
		}
	}
	*/

	if(!params.no_gui) {
		double minv,maxv;
		minMaxLoc(horiz,&minv,&maxv);
		imshow("tmp",(horiz - minv) / (maxv - minv));
		minMaxLoc(vert,&minv,&maxv);
		imshow("tmp1",(vert - minv) / (maxv - minv));
		waitKey(params.wait_time);
	}
}

template<typename T>
void HeadExtractor::getEdgesUsingTextons(Mat& lables, Mat& descriptors, Mat_<T>& horiz, Mat_<T>& vert) {
	horiz.create(lables.size()); horiz.setTo(0);	
	vert.create(lables.size()); vert.setTo(0);
	
	double sigma = 0.05;
	double two_sigma_sq = sigma * sigma * 2.0;

//	uchar* l_ptr_p1;
//	double* vert_ptr_p1;
	uchar* l_ptr_m1;
//	double* vert_ptr_m1;

	for(int y=0;y<lables.rows;y++) {
		uchar* l_ptr = lables.ptr<uchar>(y);

		if(y>0) {
			l_ptr_m1 = lables.ptr<uchar>(y-1);
		}

		for(int x=0;x<lables.cols;x++) {
			if(x>0) {
				if(l_ptr[x] == l_ptr[x-1]) { //label is similar = low energy
					horiz[y][x-1] = 1.0;
				} else { //compare the descriptors of horizintal neighbor pixels
					double d = cv::norm(descriptors.row(l_ptr[x]),descriptors.row(l_ptr[x-1]));
					horiz[y][x-1] = (exp(-(d*d)/two_sigma_sq));
				}
			}
			if(y>0) {
				if(l_ptr[x] == l_ptr_m1[x]) {
					vert[y-1][x] = 1.0;
				} else { // compare decriptors of vertical neighbor pixels
					double d = cv::norm(descriptors.row(l_ptr[x]),descriptors.row(l_ptr_m1[x]));
					vert[y-1][x] = (exp(-(d*d)/two_sigma_sq));
				}
			}
		}
	}
	if(!params.no_gui) {
		double minv,maxv;
		minMaxLoc(horiz,&minv,&maxv);
		imshow("tmp",(horiz - minv) / (maxv - minv));
		minMaxLoc(vert,&minv,&maxv);
		imshow("tmp1",(vert - minv) / (maxv - minv));
		waitKey(params.wait_time);
	}
}

void HeadExtractor::create2DGaussian(Mat& im, double sigma_x, double sigma_y, Point mean) {
	double sig_x_sq = sigma_x*sigma_x;
	double sig_y_sq = sigma_y*sigma_y;
	for(int y=0;y<im.rows;y++) {
		double* y_ptr = im.ptr<double>(y);
		for(int x=0;x<im.cols;x++) {
			//g(x,y) = exp(-((x-x0)^2)/(sigma_x^2) + ((y-y0)^2)/(sigma_x^2)))

			y_ptr[x] = exp(-((x-mean.x)*(x-mean.x)/(sig_x_sq) + (y-mean.y)*(y-mean.y)/(sig_y_sq)));
		}
	}
}

//void matting(Mat& mask, Mat& im) {
//	Mat inner;
//	erode(mask,inner,Mat::ones(10,10,CV_8UC1));
//	Mat outer;
//	dilate(mask,outer,Mat::ones(10,10,CV_8UC1));
//
//	//vector
//}

void HeadExtractor::NaiveRelabeling(Size s, vector<Mat>& backP, vector<Mat>& maskA) {
	//"naive" re-labeling using histograms backprojection maximum value
	for(int _x=0;_x<s.width;_x++) {
		for(int _y=0;_y<s.height;_y++) {
			double bp0 = backP[0].at<double>(_y,_x);
			double bp1 = backP[1].at<double>(_y,_x);
			double bp2 = backP[2].at<double>(_y,_x);

			if(fabs(bp0-bp1) < .2 && fabs(bp0-bp2) < .2 && fabs(bp1-bp2) < .2) continue;

			if(bp0 > bp1) {
				if(bp0 > bp2) {
					//0 is max
					maskA[0].ptr<uchar>(_y)[_x] = 255;
					maskA[1].ptr<uchar>(_y)[_x] = 0;
					maskA[2].ptr<uchar>(_y)[_x] = 0;
				} else {
					//2 is max
					maskA[0].ptr<uchar>(_y)[_x] = 0;
					maskA[1].ptr<uchar>(_y)[_x] = 0;
					maskA[2].ptr<uchar>(_y)[_x] = 255;
				}
			} else {
				if(bp1 > bp2) {
					//1 is max
					maskA[0].ptr<uchar>(_y)[_x] = 0;
					maskA[1].ptr<uchar>(_y)[_x] = 255;
					maskA[2].ptr<uchar>(_y)[_x] = 0;
				} else {
					//2 is max
					maskA[0].ptr<uchar>(_y)[_x] = 0;
					maskA[1].ptr<uchar>(_y)[_x] = 0;
					maskA[2].ptr<uchar>(_y)[_x] = 255;
				}
			}
		}
	}
}

int HeadExtractor::head_extract_main(int argc, char** argv) {
	//VIRTUAL_SURGEON_PARAMS params;

	params.ParseParams(argc,(const char**) argv);

	Mat im;

	if(params.filename.substr(0,6).compare("http://")) {
		params.FaceDotComDetection(im);
	} else {
		im = imread(params.filename);

		params.li = Point(253*0.4122,338*0.2624);
		params.ri = Point(253*0.5129,338*0.2603);
	}

	ExtractHead(im);

	return 0;
}

void HeadExtractor::CreateEllipses(Mat& im, Mat &maskFace, Mat &hairMask, Point& hairEllipse) {
	Point midp = (params.li + params.ri) * 0.5;
	double li_ri = -1.0;
	{
		int xdf = (params.li.x - params.ri.x);
		int ydf = (params.li.y - params.ri.y);
		li_ri = sqrt((double)(xdf*xdf) + (double)(ydf*ydf)) / (double)(im.cols);
	}

	//inner face estimate (ellipse)
	Point2d faceEllipse(
			((double)(params.li.x+params.ri.x))/2.0 - li_ri * params.yaw * 1.75,
			((double)(params.li.y+params.ri.y))/2.0 + (int)(li_ri * (double)im.cols / 2.5)
		);
	ellipse(maskFace,
		faceEllipse,
		//midp,
		Size((int)floor((double)(im.cols) * li_ri * 1/* - li_ri * params.yaw * 2.0*/),(int)floor(((double)im.cols) * li_ri * 1.68)),
		-params.roll,	//angle
		0.0,	//start angle
		360.0,	//end angle
		Scalar(255),CV_FILLED);

	if(params.gc_iter > 0)
		params.face_grab_cut(im,maskFace,1,170.0*li_ri);

	hairEllipse = Point(
			(int)floor((double)midp.x - li_ri * 6.0 * params.yaw),
			(int)floor((double)midp.y - li_ri * 5.0 * params.pitch)
			);
	ellipse(hairMask,
		//Point((li.x+ri.x)/2.0,(li.y+ri.y)/2.0 + (int)(li_ri * (double)im.rows / 3.0)),
		hairEllipse,
		Size((int)floor((double)im.cols * li_ri * 1.38 * params.hair_ellipse_size_mult),(int)floor((double)im.cols * li_ri * 1.9 * params.hair_ellipse_size_mult)),
		-params.roll,	//angle
		0.0,	//start angle
		360.0,	//end angle
		Scalar(255),CV_FILLED);

	if(!params.no_gui) {
		{
			vector<Mat> ims; split(im,ims);
			ims[0] = ims[0] + (hairMask - maskFace) * 0.5;
			ims[1] = ims[1] + maskFace * 0.5;
			//ims[2] = ims[2] & hairMask;
			Mat _im; cv::merge(ims,_im);
			imshow("tmp1",_im);
		}
		waitKey(params.wait_time);	
	}
}

void HeadExtractor::calcSegmentsLikelihood(Mat& labled_im, 
										   const vector<Mat>& masks, 
										   int bins,
										   GCoptimizationGridGraph* gc,
										   InputArray vert_edge_score,
										   InputArray horiz_edge_score,
										   int* score_matrix,
										   Mat_<char>& hard_constraints) {

	double t = (double)getTickCount();
	
	int threechans[3] = {0,1,2};
	int __histSize[3] = {bins,bins,bins};
    float __range[2] = { 0, 256.0f };
    float* __ranges[] = { __range, __range, __range };

	int zero = 0;
	int _histSize[1] = {bins};
    float _range[2] = { 0, (float)bins };
    float* _ranges[] = { _range };

	int* channels; 
	int* histSize;
	float* range;
	float** ranges;

	if(params.do_kmeans) {
		channels = &zero;
		histSize = _histSize;
		range = _range;
		ranges = _ranges;
	} else {
		channels = threechans;
		histSize = __histSize;
		range = __range;
		ranges = __ranges;
	}

	int num_dims = ((params.do_kmeans)?1:3);

	//get 1D lables frequencies from labled image
	vector<MatND> hists_nd(masks.size());
	vector<Mat> hists(masks.size());
	
	#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<masks.size();i++) {
		calcHist(
			(const Mat*)(&labled_im),
			1,
			(const int*)channels,
			(const Mat&)(masks[i]),
			hists_nd[i],
			num_dims,
			(const int*)histSize,
			(const float**)ranges,
			true,false);

		if(params.do_kmeans) {
			Mat(hists_nd[i]).copyTo(hists[i]);

			double sum = cv::sum(hists[i])[0];
			hists[i] = hists[i] / sum;

			if(!params.no_gui) {
				double maxv;
				minMaxLoc(hists[i],0,&maxv);

				Mat histP(100,bins*10,CV_8UC1,Scalar(0));
				for(int j=0;j<bins;j++) {
					cv::rectangle(histP,
						Point(j*10,100.0-((float*)hists[i].data)[j] / maxv * 100.0),
						Point(j*10+10,100),
						Scalar(255),CV_FILLED);
				}
				imshow("tmp",histP);
				imshow("tmp1",masks[i]);
				waitKey(params.wait_time);
			}
		} else {
			normalize(hists_nd[i],hists_nd[i]);
		}
	}

	//4-connectivity frequencies
	vector<Mat> segment_neighbour_probs(masks.size());
	for(int i=0;i<masks.size();i++) { 
		segment_neighbour_probs[i] = 
			Mat::zeros(
				bins * num_dims,
				bins * num_dims,
				CV_32SC4); 
	}

#define UP_PXL 0
#define RG_PXL 1
#define BT_PXL 2
#define LF_PXL 3

	if(params.consider_pixel_neighbourhood) {
		//get frequencies from labled image

		if(params.do_kmeans) {
			//every pixel is a label number
			#pragma omp parallel for schedule(dynamic)
			for(int y=0;y<labled_im.rows;y++) {
				uchar* row_ptr = labled_im.ptr<uchar>(y);
				uchar* row_m1_ptr = NULL; if(y>0) row_m1_ptr = labled_im.ptr<uchar>(y-1);
				uchar* row_p1_ptr = NULL; if(y<labled_im.rows-1) row_p1_ptr = labled_im.ptr<uchar>(y+1);

				for(int x=0;x<labled_im.cols;x++)  {
					//scan for each segment
					for(int i=0;i<masks.size();i++) {
						if(masks[i].at<uchar>(y,x) > 0) {
							uchar pxl = row_ptr[x];
							//left pixel
							if(x>0) {
								uchar pxl_t = row_ptr[x-1];
								segment_neighbour_probs[i].at<Vec4i>(pxl,pxl_t)[LF_PXL]++;
							}
							//right pixel
							if(x<labled_im.cols-1) {
								uchar pxl_t = row_ptr[x+1];
								segment_neighbour_probs[i].at<Vec4i>(pxl,pxl_t)[RG_PXL]++;
							}
							//up pixel
							if(y>0) {
								uchar pxl_t = row_m1_ptr[x];
								segment_neighbour_probs[i].at<Vec4i>(pxl,pxl_t)[UP_PXL]++;
							}
							//bottom pixel
							if(y<labled_im.rows-1) {
								uchar pxl_t = row_p1_ptr[x];
								segment_neighbour_probs[i].at<Vec4i>(pxl,pxl_t)[BT_PXL]++;
							}
						}
					}
				}
			}
		} else {
			//every pixel is RGB (or similar colorspace)
			assert(labled_im.step == labled_im.cols*3);

			for(int y=0;y<labled_im.rows;y++) {
				Vec3b* row_ptr = labled_im.ptr<Vec3b>(y);
				Vec3b* row_m1_ptr = NULL; if(y>0) row_m1_ptr = labled_im.ptr<Vec3b>(y-1);
				Vec3b* row_p1_ptr = NULL; if(y<labled_im.rows-1) row_p1_ptr = labled_im.ptr<Vec3b>(y+1);

				for(int x=0;x<labled_im.cols;x++)  {
					//scan for each segment
					for(int i=0;i<masks.size();i++) {
						if(masks[i].at<uchar>(y,x) > 0) {
							Vec3b pxl = row_ptr[x];
							//left pixel
							if(x>0) {
								Vec3b pxl_t = row_ptr[x-1];
								segment_neighbour_probs[i].at<Vec4i>(pxl[0],pxl_t[0])[LF_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[1]+bins,pxl_t[1]+bins)[LF_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[2]+bins*2,pxl_t[2]+bins*2)[LF_PXL]++;
							}
							//right pixel
							if(x<labled_im.cols-1) {
								Vec3b pxl_t = row_ptr[x+1];
								segment_neighbour_probs[i].at<Vec4i>(pxl[0],pxl_t[0])[RG_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[1]+bins,pxl_t[1]+bins)[RG_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[2]+bins*2,pxl_t[2]+bins*2)[RG_PXL]++;
							}
							//up pixel
							if(y>0) {
								Vec3b pxl_t = row_m1_ptr[x];
								segment_neighbour_probs[i].at<Vec4i>(pxl[0],pxl_t[0])[UP_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[1]+bins,pxl_t[1]+bins)[UP_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[2]+bins*2,pxl_t[2]+bins*2)[UP_PXL]++;
							}
							//bottom pixel
							if(y<labled_im.rows-1) {
								Vec3b pxl_t = row_p1_ptr[x];
								segment_neighbour_probs[i].at<Vec4i>(pxl[0],pxl_t[0])[BT_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[1]+bins,pxl_t[1]+bins)[BT_PXL]++;
								segment_neighbour_probs[i].at<Vec4i>(pxl[2]+bins*2,pxl_t[2]+bins*2)[BT_PXL]++;
							}
						}
					}
				}
			}
		}
	}

	vector<Scalar> sums(masks.size());
	for(int i=0;i<masks.size();i++) { 
		//vector<Mat> split(4);
		//cv::split(segment_neighbour_probs[i],split);
		//for(int j=0;j<4;j++)
		//	sums[i][j] = cv::sum(split[j])[0];

		//Mat _tmp32f; segment_neighbour_probs[i].convertTo(_tmp32f,CV_32F);
		sums[i] = cv::sum(segment_neighbour_probs[i]);
		//segment_neighbour_probs[i] = _tmp32f.mul(; 
	}

	float bin_quant = (float)bins / 256.0f;

	vector<Mat> probIms;
	for (int i=0;i<masks.size();i++)
		probIms.push_back(Mat(labled_im.size(),CV_32FC1));

	assert(probIms.size() >= masks.size());
	assert(probIms[0].rows >= labled_im.rows && probIms[0].cols >= labled_im.cols);
	
	//fill smoothness and data cost in graph
//	#pragma omp parallel for schedule(dynamic) shared(gc)
	for(int y=0;y<labled_im.rows;y++) {
		uchar* row_ptr = labled_im.ptr<uchar>(y);
		uchar* row_m1_ptr = NULL; if(y>0) row_m1_ptr = labled_im.ptr<uchar>(y-1);
		uchar* row_p1_ptr = NULL; if(y<labled_im.rows-1) row_p1_ptr = labled_im.ptr<uchar>(y+1);

		Vec3b* row_ptr_nd = labled_im.ptr<Vec3b>(y);
		Vec3b* row_m1_ptr_nd = NULL; if(y>0) row_m1_ptr_nd = labled_im.ptr<Vec3b>(y-1);
		Vec3b* row_p1_ptr_nd = NULL; if(y<labled_im.rows-1) row_p1_ptr_nd = labled_im.ptr<Vec3b>(y+1);

//		#pragma omp parallel for schedule(dynamic)
		for(int x=0;x<labled_im.cols;x++)  {
			int pos = y * labled_im.cols + x;

//			#pragma omp parallel for schedule(dynamic)
			for(int i=0;i<masks.size();i++) {
				double p;
				
				//allow for hard constraints: if a hard constraint is set, all probablities are 0 except for the specific label
				if(hard_constraints(y,x) != -1) {
					
					if(hard_constraints(y,x) == i)
						p = 1.0;
					else
						p = 0.0;
					
				} else { //no hard constraint on this pixel
					//data cost: P(label = i | pixel value, frequencies)
					//just get the normalized i'th-histogram value for this pixel value
					uchar pxl = row_ptr[x];
					
					if(params.do_kmeans)
						p = (double)(((float*)(hists[i].data))[pxl]);
					else {
						Vec3b pxl_nd = row_ptr_nd[x];
						p = (double)(hists_nd[i].at<float>(
										pxl_nd[0] * bin_quant,
										pxl_nd[1] * bin_quant,
										pxl_nd[2] * bin_quant));
					}

					if(params.consider_pixel_neighbourhood) {
						//smoothness cost
						Mat pxl_row = segment_neighbour_probs[i](Range(pxl,pxl+1),Range(0,bins * num_dims));
						Scalar summ = cv::sum(pxl_row);

						//to left
						if(x>0) {
							uchar pxl_t = row_ptr[x-1];
							double h_val = (double)(pxl_row.at<Vec4i>(0,pxl_t)[LF_PXL]);
							p *= h_val / summ[LF_PXL];
						}
						//right pixel
						if(x<labled_im.cols-1) {
							uchar pxl_t = row_ptr[x+1];
							double h_val = (double)(pxl_row.at<Vec4i>(0,pxl_t)[RG_PXL]);
							p *= h_val / summ[RG_PXL];
						}
						//up pixel
						if(y>0) {
							uchar pxl_t = row_m1_ptr[x];
							double h_val = (double)(pxl_row.at<Vec4i>(0,pxl_t)[UP_PXL]);
							p *= h_val / summ[UP_PXL];
						}
						//bottom pixel
						if(y<labled_im.rows-1) {
							uchar pxl_t = row_p1_ptr[x];
							double h_val = (double)(pxl_row.at<Vec4i>(0,pxl_t)[BT_PXL]);
							p *= h_val / summ[BT_PXL];
						}
					}
				}
				
//				#pragma omp critical
//				{
//					probIms[i].at<float>(y,x) = p;
//				}

				double log_l;
				if(p <= 0.0 || p > 1.0 || p != p) log_l = 10000.0f;
				else log_l = -log(p);
//				#pragma omp critical
				{
					gc->setDataCost(pos,i,log_l);
				}
			}
		}
	}

	if(!params.no_gui) {
		for(int i=0;i<masks.size();i++) {
			double maxv;
			minMaxLoc(probIms[i],0,&maxv);
			probIms[i] = probIms[i] / maxv; 
			imshow("tmp",probIms[i]);
			waitKey(params.wait_time);
		}
	}
	
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "calcSegmentsLikelihood: " << t <<"s"<< endl;
}

void HeadExtractor::MakeMaskFromCurve(Mat_<Point2f>& curve, Mat& maskFace, Mat& hairMask, Mat& backMask, Mat& backMask1) const {
	//face mask from polyline
	Mat_<Point2f> head_curve = curve(Range(100,135),Range(0,1));
	Mat_<Point> _pts; head_curve.convertTo(_pts,CV_32SC2);
	maskFace.setTo(Scalar(0));
	cv::fillPoly(maskFace,(const Point**)&_pts.data,&_pts.rows,1,Scalar(255));

	//hair mask from polyline
	head_curve = curve(Range(67,100),Range(0,1));
	head_curve.convertTo(_pts,CV_32SC2);
	hairMask.setTo(Scalar(0));
	cv::fillPoly(hairMask,(const Point**)&_pts.data,&_pts.rows,1,Scalar(255));
	hairMask = hairMask & ~maskFace; //hair area contains face, so remove it

	backMask = ~(hairMask | maskFace);

	if(params.do_two_back_kernels) {
		vector<Point> ptsv;
		for(int i=1;i<23;i++) ptsv.push_back(curve(i));
		ptsv.push_back(curve(61));
		ptsv.push_back(curve(62));
		ptsv.push_back(curve(63));
		ptsv.push_back(curve(66));
		ptsv.push_back(curve(65));
		ptsv.push_back(curve(64));
		for(int i=38;i<61;i++) ptsv.push_back(curve(i));

//		for(int i=0;i<ptsv.size();i++) cout << ptsv[i] << "\n";

		_pts = Mat(ptsv);
		backMask1.setTo(Scalar(0));
		cv::fillPoly(backMask1,(const Point**)&_pts.data,&_pts.rows,1,Scalar(255));
		backMask1 = backMask1 & backMask; //remove overlap from face+hair 
		backMask = ~(hairMask | maskFace | backMask1);
	}
}
	
	void HeadExtractor::CurveUserAidOpenCV() {
		
		/***************** aid with head curve fitting *******************
		 {
		 bool should_redraw = true;
		 namedWindow("user interaction");
		 MouseEvent mouse;
		 setMouseCallback("user interaction",onMouse,&mouse);
		 int selected_pt = -1; 
		 Point2f orig_mouse_pt; 
		 Mat_<Point2f> W(135,1);
		 for	(;;) {
		 bool dragging = (mouse.buttonState & CV_EVENT_FLAG_LBUTTON) != 0;
		 
		 if(  mouse.event == CV_EVENT_LBUTTONDOWN )
		 {
		 orig_mouse_pt = mouse.pt;
		 
		 // find closest point (speed-up: voronoi?)
		 Mat_<Point2f> sub = curve - Scalar(orig_mouse_pt.x, orig_mouse_pt.y);
		 float min = FLT_MAX;
		 for(int i=0;i<135;i++) {
		 float _n = norm(sub(i)); 
		 if(_n < min) {min = _n; selected_pt=i;}
		 }
		 cout << "selected " << selected_pt << endl;
		 
		 // weights vector
		 sub = curve - Scalar(orig_mouse_pt.x,orig_mouse_pt.y);
		 for(int i=0;i<135;i++) {
		 float _f = MIN(1.0f,10.0/norm(sub(i)));
		 W(i).x = _f;
		 W(i).y = _f;
		 }
		 }
		 if( mouse.event == CV_EVENT_LBUTTONUP) { selected_pt = -1; orig_mouse_pt.x = orig_mouse_pt.y = -1; }
		 if(dragging && orig_mouse_pt.x >= 0) {
		 Point2f pt = mouse.pt;
		 if(norm(pt - orig_mouse_pt) > 2.0) {
		 // deform curve based on weights
		 Mat_<Point2f> V = repeat((Mat_<Point2f>(1,1) << pt - orig_mouse_pt),135,1);
		 curve = curve + V.mul(W);
		 orig_mouse_pt = pt;
		 
		 should_redraw = true;
		 }
		 }
		 
		 //redraw
		 if(should_redraw){
		 Mat _curve_out; im.copyTo(_curve_out);
		 
		 MakeMaskFromCurve(curve, maskFace, hairMask, backMask, backMask1);
		 
		 vector<Mat> v(3); v[0] = hairMask; v[1] = maskFace;
		 if(params.do_two_back_kernels) {
		 v[2] = backMask + backMask1 * 0.5;
		 } else 
		 v[2] = backMask;
		 vector<Mat> imsplit; split(_curve_out,imsplit);
		 v[0] = v[0] * 0.2 + imsplit[0] * 0.8;
		 v[1] = v[1] * 0.2 + imsplit[1] * 0.8;
		 v[2] = v[2] * 0.2 + imsplit[2] * 0.8;
		 cv::merge(v,_curve_out);
		 
		 for(int i=0;i<135;i++) {
		 circle(_curve_out,*(curve[i]),3,Scalar(0,255,255),CV_FILLED);
		 }
		 
		 putText(_curve_out,"Roughly fit the head curve",Point(10,27),CV_FONT_HERSHEY_PLAIN,1.8,Scalar(0,0,255),2);
		 
		 imshow("user interaction",_curve_out);
		 should_redraw = false;
		 }
		 
		 int c = waitKey(30);
		 if( c == 'q' || c == 'Q' || c == ' ' )
		 {
		 break;
		 }
		 }
		 }/* End user aid with curve */
	}

	void SegmentationUserAidOpenCV() {
		/***************************** segmentation user aid ********************************
		 MouseEvent mouse;
		 setMouseCallback("user interaction", onMouse, &mouse);
		 bool b_ = false, f_ = false, h_ = false, change = true;
		 Point last;
		 int sizeOfBrush = 15;
		 for(;;) {
		 bool dragging = (mouse.buttonState & CV_EVENT_FLAG_LBUTTON) != 0;
		 bool hovering = (mouse.event == 0 && mouse.buttonState == 0);
		 
		 if(  mouse.event == CV_EVENT_LBUTTONDOWN ||
		 mouse.event == CV_EVENT_LBUTTONUP ||
		 dragging)
		 {
		 Point m = mouse.pt;
		 if(last!=m) {
		 if(h_ || f_ || b_) {
		 for(int i=0;i<maskA.size();i++) {
		 circle(maskA[i],m,sizeOfBrush,Scalar(0),CV_FILLED);
		 }
		 circle(hard_constraints,m,sizeOfBrush,Scalar(-1),CV_FILLED);
		 
		 if(h_) { circle(maskA[0],m,sizeOfBrush,Scalar(255),CV_FILLED); circle(hard_constraints,m,sizeOfBrush,Scalar(0),CV_FILLED);}
		 if(f_) { circle(maskA[1],m,sizeOfBrush,Scalar(255),CV_FILLED); circle(hard_constraints,m,sizeOfBrush,Scalar(1),CV_FILLED);}
		 if(b_) { circle(maskA[2],m,sizeOfBrush,Scalar(255),CV_FILLED); circle(hard_constraints,m,sizeOfBrush,Scalar(2),CV_FILLED);}
		 change = true;
		 last = m;
		 }
		 }
		 }
		 
		 //Redraw
		 if(change || hovering){
		 vector<Mat> v;  split(im_small,v);
		 //if (params.do_two_segments) {
		 //	v[0] = Mat::zeros(maskA[0].size(),CV_8UC1);
		 //}
		 int _ii;
		 for(_ii = 0; _ii < ((params.do_two_segments)?2:3); _ii++) {
		 int ind = _ii + ((params.do_two_segments)?1:0);
		 v[ind] = v[ind] * 0.7 + maskA[_ii] * 0.3;
		 }
		 if(params.do_two_back_kernels) {
		 v[1] = v[1] + maskA[_ii] * 0.15;
		 v[2] = v[2] + maskA[_ii] * 0.15;
		 }
		 
		 cv::merge(v,currentSegState);
		 
		 //if hovering - show the brush
		 {
		 circle(currentSegState, mouse.pt, sizeOfBrush, Scalar((h_)?255:0,(f_)?255:0,(b_)?255:0), 1);
		 }
		 
		 stringstream ss; 
		 ss << "Scribble to adjust ('H'air,'F'ace,'B'ack)";
		 
		 putText(currentSegState,ss.str(),Point(10,27),CV_FONT_HERSHEY_PLAIN,1.8,Scalar(0,0,255),2);
		 
		 imshow("user interaction",currentSegState);
		 }
		 
		 if(change) {	//calculate the regions on the fly
		 calcSegmentsLikelihood(tmp,maskA,params.km_numc,&gc,noArray(),noArray(),NULL,hard_constraints);
		 
		 printf("\nBefore optimization energy is %f",gc.compute_energy());
		 gc.expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		 //printf("\nAfter exp optimization energy is %d",gc.compute_energy());
		 //printf("\nBefore optimization energy is %d",gc.compute_energy());
		 //gc.swap();// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		 printf("\nAfter swap optimization energy is %f",gc.compute_energy());
		 
		 for ( int  i = 0; i < lables.rows; i++ )
		 ((int*)(lables.data + lables.step * i))[0] = gc.whatLabel(i);
		 
		 //get masks out of lables
		 for(int _ilb=0;_ilb < num_lables;_ilb++) {
		 Mat __labels = (lables == _ilb);
		 __labels = __labels.reshape(1,tmp.rows);
		 __labels.copyTo(maskA[_ilb]);
		 }
		 
		 //TODO: optimize this biggest component choosing
		 if (!params.do_two_segments) {
		 Mat _combinedHairAndFaceMask = maskA[0] | maskA[1];
		 VirtualSurgeon::takeBiggestCC(_combinedHairAndFaceMask,bias);
		 maskA[0] = maskA[0] & _combinedHairAndFaceMask; //hair
		 //maskA[1] = maskA[1] & _combinedHairAndFaceMask;
		 takeBiggestCC(maskA[1],bias);	//face
		 } else {
		 takeBiggestCC(maskA[0],bias);
		 }
		 
		 //back mask is derived from hair and face
		 if (!params.do_two_segments) {
		 maskA[2] = Mat(maskA[2].rows,maskA[2].cols,CV_8UC1,Scalar(255)) - maskA[0] - maskA[1];
		 } else {
		 maskA[1] = Mat(maskA[1].rows,maskA[1].cols,CV_8UC1,Scalar(255)) - maskA[0];
		 }
		 }
		 
		 
		 mouse.event = -1;
		 change = false;
		 
		 int c = waitKey(1);
		 if( c == 'q' || c == 'Q' || c == ' ' )
		 {
		 break;
		 }
		 if(c=='b'||c=='B') { b_ = true; f_ = false; h_ = false; }
		 if(c=='f'||c=='F') { b_ = false; f_ = true; h_ = false; }
		 if(c=='h'||c=='H') { b_ = false; f_ = false; h_ = true; }
		 if(c=='1') { sizeOfBrush = 5; }
		 if(c=='2') { sizeOfBrush = 10; }
		 if(c=='3') { sizeOfBrush = 15; }
		 }
		 
*/
	}
	
Mat HeadExtractor::ExtractHead(Mat& im, Rect r, Mat* skinMaskOut) {
	int btm_wait_time = params.wait_time;

	//if(!params.no_gui) {
	//	namedWindow("tmp");
	//	namedWindow("tmp1");
	//	namedWindow("tmp2");	
	//}
	
	Mat maskFace = Mat::zeros(im.rows,im.cols,CV_8UC1);

	//------------ hair segmentation --------------
	{
	double li_ri = -1.0;
	{
		int xdf = (params.li.x - params.ri.x);
		int ydf = (params.li.y - params.ri.y);
		li_ri = sqrt((double)(xdf*xdf) + (double)(ydf*ydf)) / (double)(im.cols);
	}

	Mat hairMask = Mat::zeros(im.rows,im.cols,CV_8UC1);
	Mat backMask = Mat::zeros(im.rows,im.cols,CV_8UC1);
	Mat backMask1 = Mat::zeros(im.rows,im.cols,CV_8UC1);

	//look for initial masks from head curve file
	bool found_head_curve = false;
	{
		Mat_<Point2f>& _curve = params.LoadHeadCurve(im,r);
		Mat_<Point2f> curve; _curve.copyTo(curve);

		StartCurveWindow(im, curve, maskFace, hairMask, backMask, backMask1, params);
		
		curve.copyTo(_curve); //changes go back to the "repo"

		MakeMaskFromCurve(_curve, maskFace, hairMask, backMask, backMask1);

		if(!params.no_gui) {
			Mat _tmp; cvtColor(maskFace,_tmp,CV_GRAY2RGB);
			putText(_tmp,"Face mask from curve",Point(5,15),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255));
			imshow("maskFace",_tmp);
			Mat _tmp1; cvtColor(hairMask,_tmp1,CV_GRAY2RGB);
			putText(_tmp1,"Hair mask from curve",Point(5,15),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255));
			imshow("hairMask",_tmp1);
			Mat _tmp2; cvtColor(backMask,_tmp2,CV_GRAY2RGB);
			putText(_tmp2,"Back mask from curve",Point(5,15),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255));
			imshow("backMask",_tmp2);
			Mat _tmp3; cvtColor(backMask1,_tmp3,CV_GRAY2RGB);
			putText(_tmp3,"Back mask 1 from curve",Point(5,15),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255));
			imshow("backMask1",_tmp3);
			waitKey(params.wait_time);
			cvDestroyAllWindows();
		}
		found_head_curve = true;
	}

	//TODO: theres a bug here, because two segments get the exact same histogram distribution...

	Point hairEllipse;
	if(!found_head_curve) {
		//get initial masks from ellipses
		CreateEllipses(im, maskFace,hairMask, hairEllipse);
	}

	hairMask = hairMask - maskFace;

	//make gabors
	vector<Mat> filter_bank;
	make_gabor_bank(filter_bank,params.gb_size,params.gb_sig,params.gb_nstds,params.gb_freq,params.gb_phase,params.gb_gamma);

	Mat im_small;
//	resize(im,im_small,Size((int)floor((double)im.cols/params.im_scale_by),
//							(int)floor((double)im.rows/params.im_scale_by)));
		resize(im,im_small,Size(),1.0/params.im_scale_by,1.0/params.im_scale_by);
		
	Mat gray;
	cvtColor(im_small,gray,CV_RGB2GRAY);

	if(!params.no_gui) {
		Mat _col;
		cvtColor(gray,_col,CV_GRAY2RGB);
		circle(_col,Point((int)((double)params.li.x/params.im_scale_by),(int)((double)params.li.y/params.im_scale_by)),2,Scalar(255,0,0),CV_FILLED);
		circle(_col,Point((int)((double)params.ri.x/params.im_scale_by),(int)((double)params.ri.y/params.im_scale_by)),2,Scalar(255,0,0),CV_FILLED);
		imshow("tmp",_col);
		waitKey(params.wait_time);
	}

	Mat tmp;

	//samples matrix
	int featVecLength = im_small.rows*im_small.cols;

	Mat featureLables;
	Mat centers;

	int numOfDoG = params.num_DoG;

	if(params.do_kmeans) {
		//size of f.vec = <gb_size> gabor responses + 3 RGB val + x,y location
		Mat featureVec(featVecLength,params.gb_size + 3 + ((params.doPositionInKM)?2:0) + numOfDoG,CV_32FC1); 

		int im_ch = im_small.channels();
		for(int i=0;i<featVecLength;i++) {
			//rgb
			for(int ch = 0; ch<3; ch++) {
				((float*)(featureVec.data + i*featureVec.step))[ch] = (float)((im_small.data + i * im_ch)[ch]) / 255.0f;
			}
			if(params.doPositionInKM) {
				//x,y position
				((float*)(featureVec.data + i*featureVec.step))[3] = (float)ceil(((float)(i % im_small.cols) / (float)im_small.cols) * (float)params.km_numc) / (float)params.km_numc;
				((float*)(featureVec.data + i*featureVec.step))[4] = (float)ceil(((float)(i / im_small.cols) / (float)im_small.rows) * (float)params.km_numc) / (float)params.km_numc;
			}
		}

		int _off = 3 + ((params.doPositionInKM)?2:0); //put gabor after RGB [and position]

		Mat gray32f; gray.convertTo(gray32f,CV_32FC1,1.0/255.0);

		for(int i=0;i<params.gb_size;i++) {
			tmp.setTo(Scalar(0));
			//Mat filter8bit; //(filter_bank[i].rows,filter_bank[i].cols,CV_8UC1);
			//Mat _f = ((filter_bank[i] + 1.0)/2.0);
			//_f.convertTo(filter8bit,CV_8UC1,255.0);

			//matchTemplate(gray,/*filter_bank[i]*/filter8bit,tmp,CV_TM_CCOEFF_NORMED);
			filter2D(gray32f,tmp,-1,filter_bank[i]);

			if(!params.no_gui) {
				imshow("tmp",tmp);
				waitKey(params.wait_time);
			}

			for(int ii=0;ii<featVecLength;ii++) {
				featureVec.at<float>(ii,_off + i) = ((float*)(tmp.data))[ii];
			}
		}

		_off += params.gb_size; //put DoG after gabor
		int ks[] = {5,7,11,15,19,21};

		for(int i=0;i<numOfDoG;i++) {
			Mat G1; GaussianBlur(gray32f,G1,Size(ks[i],ks[i]),(i+1) * 1.5);
			Mat G2; GaussianBlur(gray32f,G2,Size(ks[i],ks[i]),(i+1) * 3.0);
			Mat DoG = G1 - G2;

			if(!params.no_gui) {
				imshow("tmp",DoG);
				waitKey(params.wait_time);
			}

			DoG = DoG.reshape(1,1);

			for(int ii=0;ii<featVecLength;ii++) {
				featureVec.at<float>(ii,_off + i) = ((float*)(DoG.data))[ii];
			}
		}

		printf("Start K-Means...");
		kmeans(featureVec,
			params.km_numc,
			featureLables,
			TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,250,0.0001),
			params.km_numt,
			KMEANS_PP_CENTERS,
			centers);
		printf("end\n");

		{
			Mat __tmp = featureLables.reshape(1,im_small.rows);

			if(!params.no_gui) {
				Mat __tmp1 = __tmp * 255.0 / (double)params.km_numc;
				__tmp1.convertTo(tmp,CV_8UC1);

				imshow("tmp",tmp);
				int c = waitKey(params.wait_time);
				if(c=='q') return Mat();
			}

			__tmp.convertTo(tmp,CV_8UC1);
		}

	} else {
		//No "label space", use grayscale
		//gray.reshape(1,1).copyTo(featureLables);
		//im_small.reshape(3,1).copyTo(featureLables);
		im_small.copyTo(tmp);
	}
		
		{
	//calc histogram for face area
	//Mat _maskFace,_faceBackp;
	//resize(maskFace,_maskFace,tmp.size(),0,0,INTER_NEAREST);
	////imshow("tmp2",_maskFace);

	//Mat _hairMask;
	//resize(hairMask,_hairMask,tmp.size(),0,0,INTER_NEAREST);
	//imshow("tmp2",_hairMask);
	//calcHistogramWithMask(tmp,_hairMask,km_numc);

	//calc histogram for background area
	//Mat notHair = Mat(tmp.size(),CV_8UC1,Scalar(255)) - _maskFace - _hairMask;
	//imshow("tmp2",notHair);

	//test image for area that is not face or background (== hair!)
	//if(params.doInitStep) {
	//	MatND faceHist,backHist;
	//	Mat _backBackp;

	//	vector<MatND> hists;
	//	hists.push_back(faceHist);
	//	hists.push_back(backHist);

	//	vector<Mat> masks;
	//	masks.push_back(_maskFace);
	//	masks.push_back(notHair);

	//	vector<Mat> backPs;
	//	backPs.push_back(_faceBackp);
	//	backPs.push_back(_backBackp);

	//	calcHistogramWithMask(hists,tmp,masks,(float)params.km_numc,4,0,backPs);

	//	Mat backProjD(tmp.size(),CV_64FC1,Scalar(0));

	//	double maxV,minV;
	//	minMaxLoc(_faceBackp,&minV,&maxV);
	//	_faceBackp = (_faceBackp - minV) / (maxV - minV);
	//	minMaxLoc(_backBackp,&minV,&maxV);
	//	_backBackp = (_backBackp - minV) / (maxV - minV);

	//	switch(params.com_add_type) {
	//		case 0:
	//			cv::sqrt(_faceBackp.mul(_faceBackp) + _backBackp.mul(_backBackp),backProjD); //sqrt(score_w_back*score_w_back + score_w_face*score_w_face);
	//			break;
	//		case 1:
	//			backProjD = max(_faceBackp,_backBackp); //_bpd_ptr[x] = MAX(score_w_back,score_w_face);
	//			break;
	//		case 2:
	//			backProjD = max(max(_faceBackp,_backBackp),_faceBackp+_backBackp); //_bpd_ptr[x] = MAX(MAX(score_w_back,score_w_face),score_w_face + score_w_back);
	//			break;
	//		case 3:
	//			backProjD = _faceBackp + _backBackp; //_bpd_ptr[x] = score_w_face + score_w_back;
	//	}

	//	minMaxLoc(backProjD,&minV,&maxV);
	//	backProjD = (backProjD - minV) / (maxV - minV);

	//	if(!params.no_gui) {
	//		imshow("tmp2",backProjD);
	//	}
	//	
	//	Mat backProj = (backProjD < params.com_thresh);

	//	if(!params.no_gui) {
	//		imshow("tmp",backProj);
	//		waitKey(params.wait_time);
	//	}
	//	resize(backProj,hairMask,hairMask.size());

	//	hairMask = hairMask & ~maskFace;

	//	params.face_grab_cut(im,hairMask,params.gc_iter,18);

	//	if(!params.no_gui) {
	//		imshow("tmp",hairMask);
	//		waitKey(params.wait_time);
	//	}
	//}
		}
		
	if(params.do_two_back_kernels && !found_head_curve) {
		//create two background masks, one for top area and one for bottom
		backMask = Mat(hairMask.size(),CV_8UC1,255) - hairMask - maskFace;
		backMask1 = Mat::zeros(backMask.size(),CV_8UC1);
		ellipse(backMask1,Point(backMask1.cols/2,backMask1.rows-1),
			Size(backMask1.cols*0.45,backMask1.rows-(hairEllipse.y + (int)floor((double)im.cols * li_ri * 1.9 * params.hair_ellipse_size_mult))),
			0.0,0.0,360.0,Scalar(255),CV_FILLED);
		backMask1 = backMask1 - hairMask - maskFace;
		backMask = backMask - backMask1;
	}

	//show initial mask
	if(!params.no_gui) {
		vector<Mat> v(3); hairMask.copyTo(v[0]); maskFace.copyTo(v[1]);
		if(params.do_two_back_kernels) {
			v[2] = backMask + backMask1 * 0.5;
		} else 
			backMask.copyTo(v[2]);
		vector<Mat> imsplit; split(im_small,imsplit);
		v[0] = v[0] * 0.5 + imsplit[0] * 0.5;
		v[1] = v[1] * 0.5 + imsplit[1] * 0.5;
		v[2] = v[2] * 0.5 + imsplit[2] * 0.5;
		Mat __tmp; cv::merge(v,__tmp);
		imshow("tmp",__tmp);
		waitKey(params.wait_time);
	}

	//iterative process
	{
	Mat _it_hairMask,_it_faceMask,_it_backMask,_it_backMask1;
	resize(maskFace,_it_faceMask,tmp.size(),0,0,INTER_NEAREST);
	resize(hairMask,_it_hairMask,tmp.size(),0,0,INTER_NEAREST);
	resize(backMask,_it_backMask,tmp.size(),0,0,INTER_NEAREST);
	if(params.do_two_back_kernels) resize(backMask1,_it_backMask1,tmp.size(),0,0,INTER_NEAREST);

	MatND _it_hairHist,_it_faceHist,_it_backHist,_it_backHist1;
	
	vector<MatND> histA;
	histA.push_back(_it_hairHist); 
	histA.push_back(_it_faceHist); 
	histA.push_back(_it_backHist);
	if(params.do_two_back_kernels) histA.push_back(_it_backHist1);

	vector<Mat> maskA;
	maskA.push_back(_it_hairMask); 
	if(!params.do_two_segments) {	//for 3 segments use hair-mask
		maskA.push_back(_it_faceMask); 
	} else {						//else add hair mask to face mask
		maskA[0] = maskA[0] | _it_faceMask;
		_it_faceMask.setTo(Scalar(0));
	}
	maskA.push_back(_it_backMask);

	if(params.do_two_back_kernels) maskA.push_back(_it_backMask1);

	Mat1f grayInt,grayInt1;
	//getSobels(gray,grayInt,grayInt1);

	//TODO: use different parameter to choose?
	if(!params.do_kmeans)
		getEdgesForGC(gray,grayInt,grayInt1);
	else
		getEdgesUsingTextons(tmp,centers,grayInt,grayInt1);

	int num_lables = (params.do_two_segments)? 2 : 3;
	if(params.do_two_back_kernels) num_lables += 1;

	GCoptimizationGridGraph gc(im_small.cols,im_small.rows,num_lables);

	Mat1f _m = (Mat1f::ones(num_lables,num_lables) - Mat1f::eye(num_lables,num_lables)) * 35.0;

	gc.setSmoothCostVH(_m[0],grayInt[0],grayInt1[0]);

	Mat bias(im_small.rows,im_small.cols,CV_64FC1),bias1;

	create2DGaussian(bias,li_ri*im_small.cols*2.3,li_ri*im_small.cols*2.7,
		Point(hairEllipse.x/params.im_scale_by,hairEllipse.y/params.im_scale_by));

	//bias.copyTo(bias1);
	bias = max(bias,Mat(bias.rows,bias.cols,CV_64FC1,Scalar(0.5)));
	bias1 = -bias + 1.0;
	Mat lables(im_small.cols*im_small.rows,1,CV_32SC1);

	Mat_<char> hard_constraints(im_small.size()); hard_constraints.setTo(Scalar(-1));

	for(int i=0;i<maskA.size();i++) maskA[i] = maskA[i] * 0.5;

	StartSegmentationWindow(im_small,tmp,*this,maskA,hard_constraints,lables,num_lables,bias,gc);
	
	for(int i=0;i<maskA.size();i++) maskA[i] = maskA[i] > 10; //restore masks

	resize(_it_faceMask,maskFace,maskFace.size(),0,0,INTER_NEAREST);
	resize(_it_hairMask,hairMask,hairMask.size(),0,0,INTER_NEAREST);
	//resize(_it_backMask,backMask,backMask.size(),0,0,INTER_NEAREST);

	} //end iterative proccess


	if(params.doScore) {
		Mat gt_im = imread(params.groundtruth,0);
		int groundTruth_count = countNonZero(gt_im);
		gt_im = (gt_im ^ hairMask);

		if(!params.no_gui) {
			imshow("tmp2",gt_im);
		}
		//double segscore = sum(gt_im > 0).val[0];
		int xor_count = countNonZero(gt_im);
		double segscore = (double)xor_count / (double)groundTruth_count;
		cout << "seg score: " << segscore << "(" << xor_count << "/" << groundTruth_count << ")" << endl;

		if(!params.no_gui) {
			waitKey(params.wait_time);
		}
	}

	if(skinMaskOut != NULL)
		maskFace.copyTo(*skinMaskOut);
	maskFace = maskFace | hairMask;

	takeBiggestCC(maskFace);

	if(params.gc_iter > 0) {
		params.face_grab_cut(im,maskFace,2,10);
	}

	//alpha matting
	if(params.do_alpha_matt) {
		Mat tmpMask(maskFace.rows,maskFace.cols,CV_8UC1,Scalar(0));
		int dilate_size = params.alpha_matt_dilate_size;

		//prepare trimap
		{
			Mat __tmp(maskFace.rows,maskFace.cols,CV_8UC1,Scalar(0));
			dilate(maskFace,__tmp,Mat::ones(dilate_size,dilate_size,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT);	//probably background
			tmpMask.setTo(Scalar(128),__tmp);

			erode(maskFace,__tmp,Mat::ones(dilate_size*.5,dilate_size*.5,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT); // foreground
			tmpMask.setTo(Scalar(255),__tmp);
		}

		IplImage im_ipl = im;
		IplImage tmpMask_ipl = tmpMask;
		Matting *matting = new BayesianMatting( &im_ipl, &tmpMask_ipl );
		//Matting *matting = new RobustMatting( &((IplImage)im), &((IplImage)tmpMask) );
		matting->Solve(!params.no_gui);

		Mat(matting->alphamap).copyTo(maskFace);

		//maskFace.convertTo(maskFace,CV_8UC1,255);

		delete matting;
	}

	}

//#ifdef _PI
	if(!params.no_gui) {
		imshow("tmp",maskFace);
	}
	
	Mat imMasked;
	{
		if(maskFace.type()!=CV_32FC1) 
			maskFace.convertTo(maskFace,CV_32FC1,1.0/255.0);

		Mat unMask = Mat::ones(maskFace.size(),CV_32FC1) - maskFace; //Mat(maskFace.size(),CV_8UC1,Scalar(255)) - maskFace;
		//if(!params.no_gui) {
		//	imshow("tmp2",unMask);
		//}

		//{
		//	Mat _tmp;
		//	
		//	vector<Mat> v; 
		//	v.push_back(maskFace - hair);

		//	if (params.do_two_segments) 
		//		v.push_back(Mat::zeros(maskA[0].size(),CV_8UC1));
		//	else
		//		v.push_back(hairMask);

		//	v.push_back(unMask);

		//	cv::merge(v,_tmp);
		//	imshow("tmp2",_tmp);
		//	waitKey(params.wait_time);
		//}

		vector<Mat> v;
		im.convertTo(im,CV_32FC3,1.0/255.0);
		split(im,v);
		v[0] = v[0].mul(maskFace) + (unMask);
		v[1] = v[1].mul(maskFace);
		v[2] = v[2].mul(maskFace);
		cv::merge(v,imMasked);
	}
	if(!params.no_gui) {
		imshow("tmp1",imMasked);
		waitKey(params.wait_time);
	}
//#endif

	return maskFace;
}

}//ns
//

