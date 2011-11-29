/*
 *  LandmarksWindow.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/14/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "GUICommon.h"

class LandmarksWindow : public OpenCVImageViewer {
	VirtualSurgeon::VirtualSurgeonParams& p;
public:
	LandmarksWindow(const cv::Mat& im,VirtualSurgeon::VirtualSurgeonParams& _p,const char* label = 0):
	OpenCVImageViewer(im, label),
	p(_p),selected_pt(-1),left_selected(false),mark_left(false),mark_right(false),dragging(false)
	{}
	
	virtual void draw() {
		OpenCVImageViewer::draw();
		fltk3::color(fltk3::RED);
		fltk3::pie(p.li.x-5,p.li.y-5,10,10,0,360);
		fltk3::color(fltk3::GREEN);
		fltk3::pie(p.ri.x-5,p.ri.y-5,10,10,0,360);
	}
	
	Point orig_mouse_pt; 
	int selected_pt; 
	Point mouse_pt;
	bool left_selected, mark_left, mark_right;
	bool dragging;
	
	virtual int handle(int e) {
		int res = fltk3::Widget::handle(e);
		mouse_pt.x = fltk3::event_x(); mouse_pt.y = fltk3::event_y();
		Point pt = mouse_pt;
		Point pt_orig_mouse_pt = pt - orig_mouse_pt;
		switch (e) {
			case fltk3::PUSH:
				orig_mouse_pt = mouse_pt;
				if(mark_left) { p.li = mouse_pt; mark_left = false;}
				if(mark_right) { p.ri = mouse_pt; mark_right = false;}
				left_selected = (norm(mouse_pt - p.li) < norm(mouse_pt - p.ri));
				res = 1;
				break;
			case fltk3::RELEASE:
				selected_pt = -1; orig_mouse_pt.x = orig_mouse_pt.y = -1;
				res = 1;
				break;
			case fltk3::DRAG:
				if(left_selected) 
					p.li += pt_orig_mouse_pt;
				else
					p.ri += pt_orig_mouse_pt;
				orig_mouse_pt = pt;				
				res = 1;
				break;
			default:
				break;
		}
		redraw();
		return res;
	}
};

void StartLandmarksWindow(const cv::Mat& im_bgr, VirtualSurgeon::VirtualSurgeonParams& p) {
	cv::Mat im; cvtColor(im_bgr, im, CV_BGR2RGB);
	fltk3::DoubleWindow window(im.cols,im.rows+25);
	window.begin();
		
	LandmarksWindow lw(im,p);
	lw.label("Move the markers over the left and right eyes");
	lw.labelsize(20);
	lw.align(fltk3::ALIGN_BOTTOM | fltk3::ALIGN_WRAP);
		
	window.end();
	window.show(0,NULL);
	fltk3::run();
}