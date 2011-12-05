/*
 *  OGL_OCV_common.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/21/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;

#include "GLee.h"

#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#elif defined(__linux__) || defined(__MINGW32__) || defined(WIN32)
#  define GLEW_STATIC
//#  include <GL/glew.h>
//#  include <GL/wglew.h>
#  include <GL/gl.h>
#  include <GL/glu.h>
//#  include <GL/glext.h>
#else
#  include <gl.h>
#endif

void glEnable2D();
void glDisable2D();
void copyImgToTex(Mat& _tex_img, GLuint* texID, double* _twr = 0, double* _thr = 0);
void makePow2Texture(Mat& _tex_img, GLuint* texID, double* _twr = 0, double* _thr = 0);
