/*
 *  SegmentationWindow.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/15/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "GUICommon.h"

#include "head_extractor.h"

#include <opencv2/flann/flann.hpp>

#include "GCoptimization.h"

using namespace cv;

/*
 {
 Mat currentSegState;
 
 for(int i=0;i<maskA.size();i++) maskA[i] = maskA[i] * 0.5;
 
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
		
		for(int i=0;i<maskA.size();i++) maskA[i] = maskA[i] > 10; //restore masks
		}
 */
		
class SegmentationWidget : public OpenCVImageViewer {
private:
	Point2i mouse_pt,orig_mouse_pt;
	bool b_, f_, h_, change;
	Point last;
	int sizeOfBrush;
	Mat currentSegState;
	VirtualSurgeon::HeadExtractor& he;
	vector<Mat> maskA;
	Mat_<char>& hard_constraints;
	const VirtualSurgeon::VirtualSurgeonParams& params;
	Mat& lables;
	int num_lables;
	const Mat& bias;
	Mat& tmp;
	GCoptimizationGridGraph& gc;
public:
	SegmentationWidget(const Mat& im, 
					   Mat& _tmp,
					   VirtualSurgeon::HeadExtractor& _he,
					   vector<Mat>& _maskA,
					   Mat_<char>& _hard_constraints,
					   Mat& _labels,
					   int _num_lables,
					   const Mat& _bias,
					   GCoptimizationGridGraph& _gc,
					   const char* label = 0):
	OpenCVImageViewer(im, label),
	tmp(_tmp),
	gc(_gc),
	b_(false),f_(false),h_(false),change(true),
	sizeOfBrush(15),
	he(_he),
	maskA(_maskA),
	hard_constraints(_hard_constraints),
	lables(_labels),
	params(_he.getParams()),
	num_lables(_num_lables),
	bias(_bias)
	{
		recalculateGraph();
	}
	
	void recalculateGraph()
	{	//calculate the regions on the fly
		he.calcSegmentsLikelihood(tmp,maskA,params.km_numc,&gc,noArray(),noArray(),NULL,hard_constraints);
		
		printf("\nBefore optimization energy is %f",gc.compute_energy());
		gc.expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
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
			VirtualSurgeon::takeBiggestCC(maskA[1],bias);	//face
		} else {
			VirtualSurgeon::takeBiggestCC(maskA[0],bias);
		}
		
		//back mask is derived from hair and face
		if (!params.do_two_segments) {
			maskA[2] = Mat(maskA[2].rows,maskA[2].cols,CV_8UC1,Scalar(255)) - maskA[0] - maskA[1];
		} else {
			maskA[1] = Mat(maskA[1].rows,maskA[1].cols,CV_8UC1,Scalar(255)) - maskA[0];
		}
	}
	
	void redrawOpenCVImage() {
		vector<Mat> v;  split(im,v);
		int _ii;
		for(_ii = 0; _ii < ((he.getParams().do_two_segments)?2:3); _ii++) {
			int ind = _ii + ((he.getParams().do_two_segments)?1:0);
			v[ind] = v[ind] * 0.7 + maskA[_ii] * 0.3;
		}
		if(he.getParams().do_two_back_kernels) {
			v[1] = v[1] + maskA[_ii] * 0.15;
			v[2] = v[2] + maskA[_ii] * 0.15;
		}
		
		cv::merge(v,currentSegState);
		
		//if hovering - show the brush
		{
			circle(currentSegState, mouse_pt, sizeOfBrush, Scalar((h_)?255:0,(f_)?255:0,(b_)?255:0), 1);
		}
		img = new fltk3::RGBImage(currentSegState.data,currentSegState.cols,currentSegState.rows);
	}		
	
	virtual int handle(int e) {
		int res = Widget::handle(e);
		mouse_pt.x = fltk3::event_x(); mouse_pt.y = fltk3::event_y();
		Point pt = mouse_pt;
		Point pt_orig_mouse_pt = pt - orig_mouse_pt;
		Point m = mouse_pt;
		unsigned int c = fltk3::event_key();
		switch (e) {
			case fltk3::RELEASE:
				orig_mouse_pt.x = orig_mouse_pt.y = -1;
				redrawOpenCVImage();
				res = 1;
				break;
			case fltk3::KEYUP:
				
				if( c == 'q' || c == 'Q' || c == ' ' )
				{
					parent()->hide();
				}
				if(c=='b'||c=='B') { b_ = true; f_ = false; h_ = false; }
				if(c=='f'||c=='F') { b_ = false; f_ = true; h_ = false; }
				if(c=='h'||c=='H') { b_ = false; f_ = false; h_ = true; }
				if(c=='1') { sizeOfBrush = 5; }
				if(c=='2') { sizeOfBrush = 10; }
				if(c=='3') { sizeOfBrush = 15; }
				redrawOpenCVImage();
				break;
			case fltk3::PUSH:
			case fltk3::DRAG:
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

					recalculateGraph();
					redrawOpenCVImage();
//					redraw();
				}
				orig_mouse_pt = pt;				
				res = 1;
				break;
			case fltk3::ENTER:
			case fltk3::MOVE:
				redrawOpenCVImage();
//				redraw();
				res = 1;
				break;
			default:
				break;
		}
		redraw();
		return res;
	}
};

void StartSegmentationWindow(const Mat& im_bgr, 
							 Mat& _tmp,
							 VirtualSurgeon::HeadExtractor& _he,
							 vector<Mat>& _maskA,
							 Mat_<char>& _hard_constraints,
							 Mat& _labels,
							 int _num_lables,
							 const Mat& _bias,
							 GCoptimizationGridGraph& _gc) {
	cv::Mat im; cvtColor(im_bgr, im, CV_BGR2RGB);
	fltk3::DoubleWindow window(im.cols,im.rows+100);
	window.begin();
	
	SegmentationWidget sw(im,_tmp,_he,_maskA,_hard_constraints,_labels,_num_lables,_bias,_gc);
	sw.label("Adjust the segments\nUse the [B]ackground [H]air and [F]ace keys to select the brush\n[1] [2] and [3] select the size of the brush");
	sw.labelsize(20);
	sw.align(fltk3::ALIGN_BOTTOM | fltk3::ALIGN_WRAP);
	
	window.end();
	window.show(0,NULL);
	fltk3::run();
}
