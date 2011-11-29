/*
 *  GUICommon.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/14/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>

#include "../VirtualSurgeon_Utils/VirtualSurgeon_Utils.h"

#include <fltk3/run.h>
#include <fltk3/DoubleWindow.h>
#include <fltk3/Widget.h>
#include <fltk3/draw.h>
#include <fltk3/RGBImage.h>
#include <fltk3/Box.h>
#include <fltk3/names.h> 

class OpenCVImageViewer : public fltk3::Widget {
protected:
	const Mat& im;
	Ptr<fltk3::RGBImage> img;
public:
	OpenCVImageViewer(const cv::Mat& _im,const char* label = 0):
	im(_im),
	fltk3::Widget(0, 0, _im.cols, _im.rows, label)
	{
		img = new fltk3::RGBImage(im.data,im.cols,im.rows);
	}
	
	void setImage(cv::Mat& im) {
		img = new fltk3::RGBImage(im.data,im.cols,im.rows);
	}
	
	virtual void draw() {
		img->draw(0,0);
	}
};
