#pragma once
#include "ImageEditingUtils.h"

#include <cv.h>
using namespace cv;

template<class T>
class OpenCV2ImageWrapper :
	public ImageEditingUtils::IImage
{
	Mat& m_im;
public:
	int getRGB(int x,int y) { return m_im.ptr<T>(y)[x];}
	void setRGB(int x,int y,int rgb) { m_im.at<T>(y,x) = rgb;}

	OpenCV2ImageWrapper(Mat& im):m_im(im) {};
	~OpenCV2ImageWrapper(void) {};
};
