#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"

#include "cv.h"
#include "highgui.h"
#include "ml.h"

using namespace cv;

#include <vector>
#include <iostream>
#include <limits>

using namespace std;

#define __PI 3.14159265

void graphcut(Mat& featureVec, Mat& im, int num_lables) {
	Mat lables, centers;
	cout << "Kmeans: #centers = "<<num_lables<<",#tries = 2...";
	kmeans(featureVec,
		num_lables,
		lables,
		TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,200,0.0001),
		2,
		KMEANS_PP_CENTERS,
		centers);
	cout << "Done" << endl;

	for(int i=0;i<num_lables;i++) {
		cout << "center "<<i<<": ";
		for(int j=0;j<centers.cols;j++) cout << centers.at<float>(i,j)<<",";
		cout << endl;
	}

#ifdef BTM_DEBUG
	{
	Mat _tmp = lables.reshape(1,im.rows);
	Mat _tmpUC;
	_tmp.convertTo(_tmpUC,CV_8UC1,255.0/(double)num_lables);
	imshow("tmp", _tmpUC);
	waitKey();
	}
#endif

	Ptr<GCoptimizationGridGraph> gc = Ptr<GCoptimizationGridGraph>(new GCoptimizationGridGraph(im.cols,im.rows,num_lables));

	Mat _d(featureVec.size(),CV_32FC1);

	for(int center = 0; center < num_lables; center++) {
		_d.setTo(Scalar(0));
		int count = 0;
		for(int i=0,ii=0;i<featureVec.rows;i++) {
			if(((int*)lables.data)[i] == center) {
				float* dptr = (float*)(_d.data + _d.step * (ii++));
				float* fptr = (float*)(featureVec.data + featureVec.step * i);

				for(int j=0;j<featureVec.cols;j++) {
					dptr[j] = fptr[j];
				}

				count++;
			}
		}

		Mat d = _d(Rect(0,0,_d.cols,count));

		Mat covar;
		calcCovarMatrix(d,covar,noArray(),CV_COVAR_NORMAL+CV_COVAR_ROWS,CV_32F);

		Mat icv = covar.inv();
		Mat centerRepeat;
		repeat(centers(Rect(0,center,centers.cols,1)),featureVec.rows,1,centerRepeat);
		Mat diff = featureVec - centerRepeat; //difference between each pixel's value and it's center's value

		Mat A = (diff*icv).mul(diff) * 0.5f;
		for(int i=0;i<A.rows;i++) {
			float* _ptr = (float*)(A.data + A.step * i);
			float cost = 0;

			for(int j=0;j<A.cols;j++) {
				cost += _ptr[j];
			}

			int icost = MAX(0,(int)floor(-log(cost)));
			gc->setDataCost(i,center+1,icost);
		}
	}

	//int *smooth = new int[num_lables*num_lables];
	//int cost;
	//for ( int l1 = 0; l1 < num_lables; l1++ )
	//	for (int l2 = 0; l2 < num_lables; l2++ ){
	//		cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
	//		//Mat _a = centers(Rect(0,l1,centers.cols,1)) - centers(Rect(0,l2,centers.cols,1));
	//		//float n = (float)norm(_a);
	//		//if(n==0)	cost = 0;
	//		//else		cost = (int)ceil(-log(n));
	//		smooth[l1+l2*num_lables] = cost;
	//	}


	//Mat gk = getGaussianKernel(3,-1,CV_32F);
	//gk = gk * gk.t();
	//cv::Sobel(gk,gk,-1,1,0);
	Mat gray; cvtColor(im,gray,CV_RGB2GRAY);
	//imshow("tmp",gray); waitKey();
	gray.convertTo(gray,CV_32FC1,1.0f/255.0f);
	//imshow("tmp",gray); waitKey();

	Mat1f grayInt,grayInt1;
	{
	Mat _tmp;
	//filter2D(gray,_tmp,CV_32F,gk);
	Sobel(gray,_tmp,-1,1,0);	//sobel for dx
	//Canny(gray,_tmp,50.0,150.0);
	_tmp = abs(_tmp);
#ifdef BTM_DEBUG
	imshow("tmp",_tmp); waitKey();
#endif
	double maxVal,minVal;
	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log((_tmp - minVal) / (maxVal - minVal),_tmp);
	_tmp = -_tmp * 0.35;

	_tmp.convertTo(grayInt,grayInt.type());
	
	//grayInt = grayInt * 5;

	//filter2D(gray,_tmp,CV_32F,gk.t());
	Sobel(gray,_tmp,-1,0,1);	//sobel for dy
	//Canny(gray,_tmp,50.0,150.0);
	_tmp = abs(_tmp);
#ifdef BTM_DEBUG
	imshow("tmp",_tmp); waitKey();
#endif
	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log((_tmp - minVal) / (maxVal - minVal),_tmp);
	_tmp = -_tmp * 0.35;
	_tmp.convertTo(grayInt1,grayInt1.type());

	//grayInt1 = grayInt1 * 5;
	}

	//// next set up spatially varying arrays V and H
	//int *V = new int[featureVec.rows];
	//int *H = new int[featureVec.rows];

	//
	//for ( int i = 0; i < featureVec.rows; i++ ){
	//	//H[i] = i+(i+1)%im.rows;
	//	//V[i] = i*(i+im.cols)%im.cols;
	//	H[i] = 1;
	//	V[i] = 1;
	//}

	Mat1f Sc = 10.0 * (Mat1f::ones(num_lables,num_lables) - Mat1f::eye(num_lables,num_lables));

	gc->setSmoothCostVH(Sc[0],grayInt[0],grayInt1[0]);
	//gc->setSmoothCost((int*)(Sc.data));

	while(true) {
		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(1);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < featureVec.rows; i++ )
			((int*)(lables.data + lables.step * i))[0] = gc->whatLabel(i);

		{
		Mat _tmp = lables.reshape(1,im.rows);
#ifdef BTM_DEBUG
		{
			Mat _tmpUC;
			_tmp.convertTo(_tmpUC,CV_8UC1,255.0/(double)num_lables);
			vector<Mat> chns; split(im,chns);
			for(unsigned int ch=0;ch<chns.size();ch++) 
			{
				chns[ch] = /*chns[ch] + */(_tmp == ch)/**0.5*/;
			}
			cv::merge(chns,_tmpUC);
			imshow("tmp", _tmpUC);
			int c = waitKey();
			if(c=='q') break;
		}
#endif
		}
	}

	//delete[] smooth;
	//delete[] V;
	//delete[] H;

	//printf("%d\n",reshaped.rows);
}

template<typename T>
void graphcut1(Mat& im, Mat& probs, Mat_<T>& dx, Mat_<T>& dy,int num_lables,Mat lables = Mat()) {
	GCoptimizationGridGraph gc(im.cols,im.rows,num_lables);

	int N = im.cols*im.rows;
	//probs = probs.reshape(1,N);
	double log2 = log(1.3);
	for(int i=0;i<N;i++) {
		double* ppt = probs.ptr<double>(i);
		for(int l=0;l<num_lables;l++) {
			int icost = MAX(0,(int)floor(-log(ppt[l])/log2));
			gc.setDataCost(i,l,icost);
		}
	}

	Mat1f Sc = 5.0 * (Mat1f::ones(num_lables,num_lables) - Mat1f::eye(num_lables,num_lables));
	//int score[9] = {0,50,50,
	//				50,0,50,
	//				50,50,0};
	gc.setSmoothCostVH(Sc[0],dx[0],dy[0]);

	lables.create(N,1,CV_8UC1);

	while(true) {
		printf("\nBefore optimization energy is %d\n",gc.compute_energy());
		gc.expansion(1);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d\n",gc.compute_energy());

		for ( int  i = 0; i < N; i++ )
			((uchar*)(lables.data + lables.step * i))[0] = gc.whatLabel(i);

		{
			Mat _tmp = lables.reshape(1,im.rows);
			{
				vector<Mat> chns(3);
				for(unsigned int ch=0;ch<chns.size();ch++) chns[ch] = (_tmp == ch);
				Mat _tmpUC; cv::merge(chns,_tmpUC);
				imshow("tmp", _tmpUC);
				cout << "Press 'q' to finish GC iterations"<<endl;
				int c = waitKey();
				if(c=='q') break;
			}
		}
	}

	
}

int selectedRange = 0;
Range rs[6] = {Range(40,164),Range(56,149),Range(192,241),Range(32,185),Range(16,169),Range(163,206)};
bool draw = false;

void on_mouse( int event, int x, int y, int flags, void* param )
{
	Mat* pm = (Mat*)param;
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: 
        {
			rs[selectedRange*2].start = min(max(0,y),pm->cols-1);
			rs[selectedRange*2+1].start = min(max(0,x),pm->rows-1);
			cout << "Begin " << x << "," << y << endl;
			draw = true;
        }
        break;
    case CV_EVENT_RBUTTONDOWN: 

        break;
    case CV_EVENT_LBUTTONUP:
        {
			rs[selectedRange*2].end = min(max(0,y),pm->cols-1);
			rs[selectedRange*2+1].end = min(max(0,x),pm->rows-1);
			cout << "End " << x << "," << y << endl;
			draw = false;
        }
        break;
    case CV_EVENT_RBUTTONUP:

        break;
    case CV_EVENT_MOUSEMOVE:
		if(draw) {
			Mat _tmp; (*((Mat*)param)).copyTo(_tmp);
			if(selectedRange!=0)
				rectangle(_tmp,Point(rs[1].start,rs[0].start),Point(rs[1].end,rs[0].end),Scalar(255,0,0),2);
			else
				rectangle(_tmp,Point(rs[1].start,rs[0].start),Point(x,y),Scalar(255,0,0),2);
			if(selectedRange!=1)
				rectangle(_tmp,Point(rs[3].start,rs[2].start),Point(rs[3].end,rs[2].end),Scalar(0,255,0),2);
			else
				rectangle(_tmp,Point(rs[3].start,rs[2].start),Point(x,y),Scalar(0,255,0),2);
			imshow("tmp1",_tmp);
		}
        break;
    }
}

void getEdges(Mat& gray, Mat& grayInt, Mat& grayInt1) {
	Mat _tmp,_tmp1,gray32f,res;
	
	gray.convertTo(gray32f,CV_32FC1,1.0/255.0);

	GaussianBlur(gray32f,gray32f,Size(11,11),0.75);

	Sobel(gray32f,_tmp,-1,1,0,3);	//sobel for dx
	//Sobel(gray32f,_tmp1,-1,1,0,3,-1.0);	//sobel for -dx
	//_tmp = abs(_tmp) + abs(_tmp1);
	//_tmp.copyTo(_tmp,(_tmp > 0.0));
	//_tmp1.copyTo(_tmp1,(_tmp1 > 0.0));
	_tmp1 = abs(_tmp); // + (_tmp1 == 0.0);
	_tmp1.copyTo(res,(_tmp1 > 0.2));
	//res = -res + 1.0;

	imshow("tmp",res);

	double maxVal,minVal;
	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log(/*(_tmp - minVal) / (maxVal - minVal)*/res,_tmp);
	_tmp = -_tmp * 0.17;
	_tmp.convertTo(grayInt1,CV_32SC1);
	
	Sobel(gray32f,_tmp,-1,0,1,3);	//sobel for dy
	//Sobel(gray32f,_tmp1,-1,0,2,3,-1.0);	//sobel for -dy
	//_tmp = abs(_tmp) + abs(_tmp1);
	//_tmp = (_tmp + _tmp1 + 2.0) / 4.0;
	_tmp1 = abs(_tmp);
	res.setTo(Scalar(0));
	_tmp1.copyTo(res,(_tmp1 > 0.2));
	//res = -res+1.0;

	imshow("tmp1",res); waitKey();

	minMaxLoc(_tmp,&minVal,&maxVal);
	cv::log(/*(_tmp - minVal) / (maxVal - minVal)*/res,_tmp);
	_tmp = -_tmp * 0.17;
	_tmp.convertTo(grayInt,CV_32SC1);

}

void doEM3D(Mat& _im, Mat& probs,int num_models = 2,int num_gaussians = 3, bool useRanges = true) {
	Mat im = _im;

	vector<CvEM> model(num_models);
	CvEMParams ps(num_gaussians);

	imshow("tmp1",im);

	vector<Mat> samples(num_models);
	while(true) {
		if(useRanges) {
			cout << "Define ranges (press <space> to continue)"<<endl;
			while(true) {
				int c = waitKey();
				if(c==' ') break;
				else selectedRange = c - '1';
				cout << "Selected range: " << selectedRange << endl;
			}
			
			vector<Mat> splitted;
			for(int i=0;i<num_models;i++) {
				Mat _tmp; im(rs[i*2],rs[i*2+1]).copyTo(_tmp);
				_tmp.reshape(1,_tmp.rows*_tmp.cols).convertTo(samples[i],CV_32FC1,1.0/255.0);
			}
		}

		for(int i=0;i<num_models;i++) {
			model[i].clear();
			model[i].train(samples[i],Mat(),ps,NULL);

			cout << "Model "<<i<<" means: ";
			const CvMat* m = model[i].get_means();
			for(int g=0;g<num_gaussians;g++) {
				for(int c=0;c<3;c++) cout << m->data.db[g*3 + c] <<", ";
				cout << endl;
			}
			cout << endl;
		}

		Mat out_lables(im.size(),CV_32FC1);
		float _s[3];
		Mat sample(1,3,CV_32FC1,&_s);
		
		probs = Mat(im.rows*im.cols,num_models,CV_64FC1);
		{
			Mat _tmp(1,2,CV_64FC1);
			for(int y=0;y<im.rows;y++) {
				uchar* imp = im.ptr<uchar>(y);
				float* outl_ptr = out_lables.ptr<float>(y);
				int probs_mult = y*im.cols;
				for(int x=0;x<im.cols;x++) {
					float _label,maxv = std::numeric_limits<float>::min();
					double p;
					int probs_mult1 = x*num_models;
					for(int i=0;i<num_models;i++) {
						float ps[3];
						for(int c=0;c<3;c++) {
							_s[c] = ((float)imp[c + x*3]) / 255.0f;
						}

						Mat X; sample.copyTo(X);
						Mat xprob;
						model[i].predict(X,&xprob);
						double* w = model[i].get_weights()->data.db;
						p = std::numeric_limits<double>::min();
						for(int g=0;g<num_gaussians;g++) p = max(p,w[g]*(double)(((float*)xprob.data)[g]));
					
						probs.at<double>(probs_mult + x,i) = p;
						if(p>maxv) { maxv = p; _label = (float)i; }
					}
					outl_ptr[x] = _label;
				}
			}
		}

		{
			Mat _tmp(im.size(),CV_8UC1);
			out_lables.convertTo(_tmp,CV_8UC1);
			vector<Mat> vm(3); for(int i=0;i<3;i++) vm[i] = (_tmp == i);
			Mat out; merge(vm,out);
			imshow("tmp",out);

			if(num_models==3)
				imshow("tmp2",probs.reshape(num_models,im.rows)); 
		}

		cout<<"Press any key for another round or 'q' to finish"<<endl;
		int c = waitKey();
		if(c=='q') break;
	}
}

/**
 * if useRanges is true, use the mouse to define ranges, else use the labeling from lables 
 * to train the GMM
 */
void doEM1D(Mat& _im, Mat& probs,int num_models = 2,bool useRanges = true,Mat lables = Mat()) {
	//Mat im; cvtColor(_im,im,CV_RGB2HSV);
	Mat im = _im;

	vector<vector<CvEM> > model(num_models);
	for(int i=0;i<num_models;i++) {
		model[i] = vector<CvEM>(3);
	}
	CvEMParams ps(1);

	imshow("tmp1",im);

	vector<vector<Mat> > samples(num_models);
	for(int i=0;i<num_models;i++) samples[i]=vector<Mat>(3);
	while(true) {
		if(useRanges) {
			cout << "Define ranges (press <space> to continue)"<<endl;
			while(true) {
				int c = waitKey();
				if(c==' ') break;
				else selectedRange = c - '1';
				cout << "Selected range: " << selectedRange << endl;
			}
			
			vector<Mat> splitted;
			for(int i=0;i<num_models;i++) {
				int cnt = 0;
				Mat _tmp; im(rs[i*2],rs[i*2+1]).copyTo(_tmp);
				//Mat __tmp = _tmp.reshape(1,_tmp.rows*_tmp.cols);
				split(_tmp,splitted);
				for(int c=0;c<3;c++)
					splitted[c].reshape(1,_tmp.rows*_tmp.cols).convertTo(samples[i][c],CV_32FC1,1.0/255.0);
			}
		}

		for(int i=0;i<num_models;i++) {
			for(int c=0;c<3;c++) {
				model[i][c].clear();
				model[i][c].train(samples[i][c],Mat(),ps,NULL);
				//m_probs[i] = model[i].get_probs();
			}

			cout << "Model "<<i<<" means: ";
			for(int c=0;c<3;c++) cout << model[i][c].get_means()->data.db[0] <<", ";
			cout << endl;
		}

		Mat out_lables(im.size(),CV_32FC1);
		float _s;
		Mat sample(1,1,CV_32FC1,&_s);
		
		probs = Mat(im.rows*im.cols,num_models,CV_64FC1);
		{
			Mat _tmp(1,2,CV_64FC1);
			for(int y=0;y<im.rows;y++) {
				uchar* imp = im.ptr<uchar>(y);
				float* outl_ptr = out_lables.ptr<float>(y);
				int probs_mult = y*im.cols;
				for(int x=0;x<im.cols;x++) {
					float _label,p,maxv = std::numeric_limits<float>::min();
					int probs_mult1 = x*num_models;
					for(int i=0;i<num_models;i++) {
						float ps[3];
						for(int c=0;c<3;c++) {
							_s = ((float)imp[c + x*3]) / 255.0f;
							//model[i][c].predict(sample,&_tmp);
							double mu = model[i][c].get_means()->data.db[0];
							double x = (double)_s;
							double sigma_sq = model[i][c].get_covs()[0]->data.db[0];
							double _p = (1/sqrt(2*__PI*sigma_sq))*exp(-((x-mu)*(x-mu))/(2*sigma_sq));
							ps[c] = (float)_p; //((double*)_tmp.data)[0];
						}
						p = ps[0]*ps[1]*ps[2];
						//((double*)probs.data + probs_mult + probs_mult1 + i)[0] = p;
						probs.at<double>(probs_mult + x,i) = p;
						if(p>maxv) { maxv = p; _label = (float)i; }
					}
					outl_ptr[x] = _label;
				}
			}
		}

		{
			Mat _tmp(im.size(),CV_8UC1);
			out_lables.convertTo(_tmp,CV_8UC1);
			vector<Mat> vm(3); for(int i=0;i<3;i++) vm[i] = (_tmp == i);
			Mat out; merge(vm,out);
			imshow("tmp",out);

			if(num_models==3)
				imshow("tmp2",probs.reshape(num_models,im.rows)); 
		}

		cout<<"Press any key for another round or 'q' to finish"<<endl;
		int c = waitKey();
		if(c=='q') break;
	}
}

int _main(int argc, char** argv) {
	Mat _im = imread("40406598_fd4e74d51c_d.jpg");
	Mat im;
	resize(_im,im,Size(_im.cols/2,_im.rows/2));

	rs[4] = Range(0,im.rows-1);
	rs[5] = Range(0,im.cols-1);

	namedWindow("tmp");
	namedWindow("tmp1");
	cvSetMouseCallback( "tmp1", on_mouse, &im );
	namedWindow("tmp2");

	Mat gray;
	cvtColor(im,gray,CV_RGB2GRAY);

	Mat probs;
	Mat hsv_im; cvtColor(im,hsv_im,CV_BGR2HSV);
	vector<Mat> v; split(hsv_im,v);

	{
		Mat gray32f; gray.convertTo(gray32f,CV_32FC1,1.0/255.0);
		Mat _tmp; Sobel(gray32f,_tmp,-1,1,1,3);
		Mat _tmp1; Sobel(gray32f,_tmp1,-1,1,1,3,-1.0);
		Mat(abs(_tmp) + abs(_tmp1)).convertTo(v[2],CV_8UC1,255.0);
	}

	Mat combined; cv::merge(v,combined);
	doEM3D(combined,probs,3);

	Mat1f dx,dy; 
	getEdges(gray,dx,dy);

	Mat lables;
	graphcut1(im,probs,dx,dy,3,lables);

	char* winnm[3] = {"tmp","tmp1","tmp2"};
	for(int i=0;i<3;i++)
	{
		Mat _tmp; im.copyTo(_tmp,lables==i);
		imshow(string(winnm[i]),_tmp);
	}
	waitKey();

	{
		Mat _tmpLabels = lables.reshape(1,im.rows);
		//find connected components in hair and face masks
		vector<vector<Point> > contours;
		for(int itr=0;itr<2;itr++) {
			Mat mask = (_tmpLabels == itr);

			contours.clear();
			cv::findContours(mask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

			//compute areas
			vector<double> areas(contours.size());
			for(unsigned int ai=0;ai<contours.size();ai++) {
				Mat _pts(contours[ai]);
				Scalar mp = mean(_pts);

				//bias score according to distance from center face
				areas[ai] = contourArea(Mat(contours[ai]))/* * bias.at<double>(mp.val[1],mp.val[0])*/;
			}

			//find largest connected component
			double max; Point maxLoc;
			minMaxLoc(Mat(areas),0,&max,0,&maxLoc);

			//draw back on mask
			_tmpLabels.setTo(Scalar(3),mask);	//all unassigned pixels will have value of 3, later we'll use it
			
			mask.setTo(Scalar(0)); //clear...
			drawContours(mask,contours,maxLoc.y,Scalar(255),CV_FILLED);

			_tmpLabels.setTo(Scalar(itr),mask);
		}

		lables.setTo(Scalar(2),lables==3);  //all 3's should be the "other label"
	}

	for(int i=0;i<3;i++)
	{
		Mat _tmp; im.copyTo(_tmp,lables==i);
		imshow(string(winnm[i]),_tmp);
	}
	waitKey();

	return 0;
}

