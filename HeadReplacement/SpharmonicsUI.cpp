/*
 *  SpharmonicsUI.cpp
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/21/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "SpharmonicsUI.h"

int SpharmonicsUI_main(int argc, char **argv, 
			   const VirtualSurgeon::VirtualSurgeonFaceData& face_data, 
			   Mat& _inputFace,
			   Mat& relitFace,
			   Mat& relitMask,
			   Rect& relitRect)
{	
	Mat inputFace; _inputFace.copyTo(inputFace);

	SpharmonicsUI window(face_data,inputFace);
	window.show();
//	window.make_current();
	fltk3::run();
	
	if (!window.spharmonics_error) {
		window.getReLitFace().copyTo(relitFace);
		window.getReLitMask().copyTo(relitMask);
		relitRect = Rect(window.getReLitRect());
	} else {
		return 1;
	}

	destroyAllWindows(); //close all OpenCV windows
	
	return 0;
}