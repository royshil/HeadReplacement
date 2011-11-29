/*
 *  CurveWindow.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/14/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */


#include "GUICommon.h"

#include "head_extractor.h"
	
#include <opencv2/flann/flann.hpp>

class CurveWidget : public OpenCVImageViewer {
	Mat_<Point2f>& curve;
	Point2i orig_mouse_pt; 
	int selected_pt; 
	Point2i mouse_pt;
	bool left_selected, mark_left, mark_right;
	bool dragging;
	Mat_<Point2f> W;
	Mat& maskFace;
	Mat& hairMask;
	Mat& backMask;
	Mat& backMask1;
	const VirtualSurgeon::HeadExtractor& he;
	
public:
	CurveWidget(const cv::Mat& im,
				Mat_<Point2f>& _curve,
				Mat& _maskFace, 
				Mat& _hairMask, 
				Mat& _backMask, 
				Mat& _backMask1,
				const VirtualSurgeon::HeadExtractor& _he,
				const char* label = 0):
	OpenCVImageViewer(im, label),
	W(135,1),
	curve(_curve),
	he(_he),
	selected_pt(-1),left_selected(false),mark_left(false),mark_right(false),dragging(false),
	maskFace(_maskFace),
	hairMask(_hairMask),
	backMask(_backMask),
	backMask1(_backMask1)
	{
		redrawOpenCVImage();
	}		
	
	void redrawOpenCVImage() {
		cv::Mat _curve_out; im.copyTo(_curve_out);
		
		he.MakeMaskFromCurve(curve, maskFace, hairMask, backMask, backMask1);
		
		vector<cv::Mat> v(3); v[0] = hairMask; v[1] = maskFace;
		if(he.getParams().do_two_back_kernels) {
			v[2] = backMask + backMask1 * 0.5;
		} else 
			v[2] = backMask;
		vector<cv::Mat> imsplit; cv::split(_curve_out,imsplit);
		v[0] = v[0] * 0.2 + imsplit[0] * 0.8;
		v[1] = v[1] * 0.2 + imsplit[1] * 0.8;
		v[2] = v[2] * 0.2 + imsplit[2] * 0.8;
		cv::merge(v,_curve_out);
		
		for(int i=0;i<135;i++) {
			cv::circle(_curve_out,*(curve[i]),3,cv::Scalar(0,255,255),CV_FILLED);
		}
		img = new fltk3::RGBImage(_curve_out.data,_curve_out.cols,_curve_out.rows);
	}		
	
	virtual int handle(int e) {
		int res = fltk3::Widget::handle(e);
		mouse_pt.x = fltk3::event_x(); mouse_pt.y = fltk3::event_y();
		cv::Point2i pt = mouse_pt;
		cv::Point2i pt_orig_mouse_pt = pt - orig_mouse_pt;
		
		if (e==fltk3::PUSH) {
			orig_mouse_pt = mouse_pt;
			if (fltk3::event_button() == fltk3::LEFT_MOUSE) {
//				vector<float> q; q.push_back(mouse_pt.x);q.push_back(mouse_pt.y);
//				vector<int> idxs(1); vector<float> dists(1);
//				Mat_<float> curve1ch = ((Mat)curve).reshape(1);
//				cv::flann::Index_<float> idx(curve1ch,cv::flann::KDTreeIndexParams());
//				
//				idx.knnSearch(q, idxs, dists, 1,cv::flann::SearchParams());
//				cout << "selected (#"<<idxs[0]<<") " << curve(idxs[0]) << endl;
				
				// weights vector
				Mat_<cv::Point2f> sub = curve - Scalar(orig_mouse_pt.x,orig_mouse_pt.y);
				for(int i=0;i<135;i++) {
					float _f = MIN(1.0f,10.0/norm(sub(i)));
					W(i).x = _f;
					W(i).y = _f;
				}
			}
			res = 1;
		} else if (e == fltk3::RELEASE) {
			selected_pt = -1; orig_mouse_pt.x = orig_mouse_pt.y = -1;
			res = 1;
		} else if (e == fltk3::DRAG) {
			if(norm(pt_orig_mouse_pt) > 2.0) {
				//cout << pt_orig_mouse_pt << endl;
				Mat_<cv::Point2f> V = repeat(Mat_<Point2f>(1,1) << Point2f(pt_orig_mouse_pt.x,pt_orig_mouse_pt.y),135,1);
				if (fltk3::event_button() == fltk3::LEFT_MOUSE) {
					// deform curve based on weights
					curve = curve + V.mul(W);
				} else if (fltk3::event_button() == fltk3::MIDDLE_MOUSE) {
					//move all points together
					curve = curve + V;
				}
				orig_mouse_pt = pt;
				redrawOpenCVImage();
				fltk3::redraw();
			}
			res = 1;
		}
		
		return res;
	}
};

void StartCurveWindow(const cv::Mat& im_bgr, 
					  Mat_<Point2f>& curve,
					  Mat& _maskFace, 
					  Mat& _hairMask, 
					  Mat& _backMask, 
					  Mat& _backMask1,
					  const VirtualSurgeon::HeadExtractor& _he) {
	cv::Mat im; cvtColor(im_bgr, im, CV_BGR2RGB);
	fltk3::DoubleWindow window(im.cols,im.rows+70);
	window.begin();
	
	CurveWidget cw(im,curve,_maskFace,_hairMask,_backMask,_backMask1,_he);
	cw.label("Roughly fit the head curve\nDrag with the mouse to squeeze/stretch the points");
	cw.labelsize(20);
	cw.align(fltk3::ALIGN_BOTTOM | fltk3::ALIGN_WRAP);
	
	window.end();
	window.show(0,NULL);
	fltk3::run();
}