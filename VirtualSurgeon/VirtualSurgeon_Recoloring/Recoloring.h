#pragma once

#include <vector>
using namespace std;

#include "../VirtualSurgeon_Utils/VirtualSurgeon_Utils.h"

namespace VirtualSurgeon {

class Recoloring
{
private:
	void TrainGMM(CvEM& source_model, Mat& source, Mat& source_mask);
	vector<int> MatchGaussians(CvEM& source_model, CvEM& target_model);

	VirtualSurgeonParams m_p;
public:
	void Recolor(Mat& source, Mat& source_mask, Mat& target, Mat& target_mask, Mat alt_source_mask = Mat());
	Recoloring(VirtualSurgeonParams& _p):m_p(_p) {};
	~Recoloring(void);
};


}//ns