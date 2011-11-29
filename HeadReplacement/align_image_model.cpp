/*
 *  align_image_model.cpp
 *  FaceTracker
 *
 *  Created by Roy Shilkrot on 10/25/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "align_image_model.h"

int align_main(int argc, char **argv, 
			   const VirtualSurgeon::VirtualSurgeonFaceData& face_data, 
			   Mat& _inputFace,
			   Mat& relitFace,
			   Mat& relitMask,
			   Rect& relitRect)
{
	Glow::Init(argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	Mat inputFace; _inputFace.copyTo(inputFace);
	Ptr<AlignFaceToModel> aftm = Ptr<AlignFaceToModel>(new AlignFaceToModel(face_data,inputFace));
	Glow::MainLoop();
	aftm->getReLitFace().copyTo(relitFace);
	aftm->getReLitMask().copyTo(relitMask);
	relitRect = Rect(aftm->getReLitRect());
	
	destroyAllWindows(); //close all OpenCV windows
	
	return 0;
}

void align_thread_func(void* ptr) {
	//	align_main(0, NULL, *(string*)ptr);
}
