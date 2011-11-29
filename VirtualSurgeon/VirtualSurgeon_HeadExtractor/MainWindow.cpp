/*
 *  MainWindow.cpp
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/22/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "MainWindow.h"
#include <opencv2/opencv.hpp>

bool male = true;
bool is_camera = true;

void male_cb(fltk3::Widget *w, void *) {
	male = true;
	w->parent()->hide();
}

void female_cb(fltk3::Widget *w, void *) {
	male = false;
	w->parent()->hide();
}

void camera_cb(fltk3::Widget *w, void *) {
	is_camera = true;
	w->parent()->hide();
}

void filebtn_cb(fltk3::Widget *w, void *) {
	is_camera = false;
	w->parent()->hide();
}

void StartMainWindow( VirtualSurgeon::VirtualSurgeonParams& p)  {
	fltk3::DoubleWindow window(330,140,"Main");
	window.begin();
	
	fltk3::Button malebtn(10,10, 150,120, "Male");
	malebtn.labelsize(40);
	malebtn.callback(male_cb, 0);
	fltk3::Button female(170,10, 150,120, "Female");
	female.labelsize(40);
	female.callback(female_cb, 0);
	
	window.end();
	window.show(0,NULL);
	fltk3::run();
	
	p.is_female = !male;

	/*
	window.remove(male);
	window.remove(female);
	window.begin();
	
	fltk3::Button camera(10,10, 150,120, "Camera");
	camera.labelsize(37);
	camera.callback(camera_cb, 0);
	fltk3::Button filebtn(170,10, 150,120, "File");
	filebtn.labelsize(40);
	filebtn.callback(filebtn_cb, 0);
	
	window.end();
	window.show(0,NULL);
	fltk3::run();
	window.remove(camera);
	window.remove(filebtn);
	
	if (is_camera) {
		cv::VideoCapture capture(CV_CAP_ANY); //try to open string, this will attempt to open it as a video file
		if (!capture.isOpened())
		{
			cerr << "Failed to open a video device or video file!\n" << endl;
			return;
		}
		Mat frame;
		bool first = true;
		Ptr<OpenCVImageViewer> imgv = NULL;
		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				continue;
			if (first) {
				first = !first;
				window.begin();
				imgv = new OpenCVImageViewer(frame);
				window.add(imgv);
				window.end();
				window.show(0,NULL);
			}
			imgv->setImage(frame);
			window.redraw();
		}
		
	}
	 */
}