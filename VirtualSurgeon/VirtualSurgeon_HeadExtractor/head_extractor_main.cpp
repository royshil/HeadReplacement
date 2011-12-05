#include "StdAfx.h"

#include "head_extractor.h"

#ifdef HEAD_EXTRACTOR_MAIN

#include "../matting/matting/matting.h"
#include "../VirtualSurgeon_Recoloring/Recoloring.h"
#include "../../HeadReplacement/LaplacianBlending.h"

#include <stdlib.h>

#include <fltk3/FileChooser.h> 
#ifdef VIRTUAL_SURGEON_DEBUG
#  include <fltk3/ask.h>
#endif

//#include <QtGui/qfiledialog.h>
//#include <QtGui/qapplication.h>

int SpharmonicsUI_main(int argc, char **argv, 
			   const VirtualSurgeon::VirtualSurgeonFaceData& face_data, 
			   Mat& inputFace,
			   Mat& relitFace,
			   Mat& relitMask,
			   Rect& relitRect);


struct MouseEvent
{
    MouseEvent() { event = -1; buttonState = 0; }
    Point pt;
    int event;
    int buttonState;
	
	friend std::ostream& operator<< (std::ostream& o, struct MouseEvent const& me) {
		return  o << "MouseEvent: ["<<me.pt<<",event="<<me.event<<",buttonState="<<me.buttonState<<"]\n";
	}
};

static void onMouse(int event, int x, int y, int flags, void* userdata)
{
    MouseEvent* data = (MouseEvent*)userdata;
    data->event = event;
    data->pt = Point(x,y);
    data->buttonState = flags;
//	cout << *data;
}

void HeadPositionUserInteraction(const Mat& face_orig_image, 
								const Mat& face_orig_mask, 
								Mat& face_area_image, 
								Mat& face_mask, 
								Rect& face_rect, 
								Mat& model_image, 
								Rect& model_rect,
								double& scaleFromFaceToBack) 
								/************************ head positioning aid ****************************/
{
	cout << "/************************ head positioning user aid ****************************/\n";
	MouseEvent mouse;
	namedWindow("user interaction");
	setMouseCallback("user interaction", onMouse, &mouse);
	Point2f orig_mouse_pt(-1,-1);
	bool scaling = false, timer_running = false;

	Mat mask32f; 
	if(face_mask.type() == CV_8UC1) face_mask.convertTo(mask32f,CV_32FC1,1.0/255.0);
	else mask32f = face_mask;
	Mat antiMask = -mask32f + 1.0;

	time_t start_t;

	double model_rect_ratio = (double)face_rect.height/(double)face_rect.width;
	cout << "model_rect_ratio " << model_rect_ratio <<endl;

	if(face_area_image.empty() || face_area_image.type() != CV_32F)
		face_orig_image(face_rect).convertTo(face_area_image,CV_32F);

	for(;;) {
		bool dragging = (mouse.buttonState & CV_EVENT_FLAG_LBUTTON) != 0;


		if(  mouse.event == CV_EVENT_LBUTTONDOWN  ) { 
			orig_mouse_pt = mouse.pt;
			if(!timer_running) { time(&start_t); timer_running = true; } //start timer on user click
		}
		if(  mouse.event == CV_EVENT_LBUTTONUP  ) { orig_mouse_pt.x = orig_mouse_pt.y = -1; dragging = false;}

		if(dragging && orig_mouse_pt.x >= 0) {
			Point2f pt = mouse.pt;
			if(scaling) { //---------------------- scaling -----------------------
				double scale_ = pt.x - orig_mouse_pt.x;
					
				//cout << pt.x << " <> " << orig_mouse_pt.x << ", scale = " << scale_ << endl;
				if(abs(scale_) >= 1.0) {
					model_rect.width += scale_;
					model_rect.height = model_rect.width * model_rect_ratio;

					scaleFromFaceToBack = (double)model_rect.width / (double)face_rect.width;
					//cout << "scaleFromFaceToBack " << scaleFromFaceToBack << endl;

					Mat _tmp_o_; 
					face_orig_image(face_rect).convertTo(_tmp_o_,CV_32F);
					resize(_tmp_o_,face_area_image,model_rect.size(),0.0,0.0,INTER_LANCZOS4);

					Mat _mask32f; resize(face_orig_mask(face_rect),_mask32f,model_rect.size()); 
					_mask32f.convertTo(mask32f,CV_32F,(face_orig_mask.type()==CV_8U) ? 1.0/255.0 : 1.0);
					antiMask = -mask32f + 1.0;
				}
			} else {	//------------------- dragging - moving ------------------
				model_rect.x -= orig_mouse_pt.x-pt.x;
				model_rect.y -= orig_mouse_pt.y-pt.y;

				//if(model_rect.x < 0 ||model_rect.y < 0) {
				//	if(model_rect.x < 0) { face_rect.x -= model_rect.x; face_rect.width += model_rect.x; model_rect.x = 0; }
				//	if(model_rect.y < 0) { face_rect.y -= model_rect.y; face_rect.height += model_rect.y; model_rect.y = 0; }
				//	{
				//		face_orig_image(face_rect).convertTo(face_area_image,CV_32F);
				//		Mat __tmp;
				//		resize(face_area_image,__tmp,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
				//		__tmp.copyTo(face_area_image);
				//		resize(face_orig_mask(face_rect),face_mask,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
				//		face_mask.convertTo(mask32f,CV_32FC1,1.0/255.0);
				//		antiMask = -mask32f + 1.0;
				//	}
				//	model_rect.width = face_area_image.cols;
				//	model_rect.height = face_area_image.rows;
				//}

//				cout << "moved: (" << model_rect.x <<","<<model_rect.y<<"),"<<model_rect.width<<","<<model_rect.height << endl;
			}
			orig_mouse_pt = pt;
		}

		//redraw
		Mat _head_pos_out; model_image.convertTo(_head_pos_out,CV_32FC3,1.0/255.0);

		Rect output_model_rect = model_rect,output_face_rect(Point(0,0),face_area_image.size());
		bool any_change = false;
		if(model_rect.x + model_rect.width > _head_pos_out.cols-1) {
			cout << "overflow on x of " << _head_pos_out.cols - model_rect.x - model_rect.width << " pixels";
			output_model_rect.width += _head_pos_out.cols - model_rect.x - model_rect.width;
			any_change = true;
		}
		if(model_rect.y + model_rect.height > _head_pos_out.rows-1) {
			output_model_rect.height += _head_pos_out.rows-1 - model_rect.y - model_rect.height;
			any_change = true;
		}
		if(model_rect.x < 0) {
			output_model_rect.width += output_model_rect.x;
			output_face_rect.x -= output_model_rect.x; //trim on left
			output_model_rect.x = 0;
			any_change = true;
		}
		if(model_rect.y < 0) {
			output_model_rect.height += output_model_rect.y;
			output_face_rect.y -= output_model_rect.y; //trim on up
			output_model_rect.y = 0;
			any_change = true;
		}

		if(any_change) {
			output_face_rect.width = output_model_rect.width;
			output_face_rect.height = output_model_rect.height;
			cout << "model rect: (" << output_model_rect.x <<","<<output_model_rect.y<<"),"<<output_model_rect.width<<","<<output_model_rect.height << endl;
			cout << "face rect: (" << output_face_rect.x <<","<<output_face_rect.y<<"),"<<output_face_rect.width<<","<<output_face_rect.height << endl;
		}

		vector<Mat> ov; cv::split(face_area_image(output_face_rect) / 255.0,ov);
		vector<Mat> mv; cv::split(_head_pos_out(output_model_rect),mv);
		for(int i=0;i<ov.size();i++) {
			ov[i] = ov[i].mul(mask32f(output_face_rect)) + mv[i].mul(antiMask(output_face_rect));
		}
		Mat out;
		cv::merge(ov,out);
		Mat _head_pos_out_output_model_rect = _head_pos_out(output_model_rect);
		out.copyTo(_head_pos_out_output_model_rect);

		//rectangle(_head_pos_out,model_rect,Scalar(255,0,0),2);
		//circle(_head_pos_out,*model_loc,8,Scalar(255,0,255),2);
		//circle(_head_pos_out(modelRect),face_p1,8,Scalar(0,255,255),CV_FILLED);
		putText(_head_pos_out,(scaling)?"Scale":"Move",Point(10,27),CV_FONT_HERSHEY_PLAIN,2.0,Scalar(0,0,255),2);

		imshow("user interaction",_head_pos_out);

		int c = waitKey(30);
		if( c == 'q' || c == 'Q' || c == ' ' || c == 27) break;
		if( c=='s' || c=='S') scaling = !scaling;
	}
	mask32f.convertTo(face_mask,CV_8UC1,255.0);

	time_t end_t; time(&end_t);
	stringstream ss; ss << difftime(end_t,start_t) << " seconds";
	putText(model_image,ss.str(),Point(10,model_image.cols -30),CV_FONT_HERSHEY_PLAIN,2.0,Scalar(0,255,255),2);	
}

void getModelImage(Mat* model_image, Mat* model_skin_mask, VirtualSurgeon::VirtualSurgeonParams* m_model_data) {
	*model_image = imread(m_model_data->filename);
	Mat tmp = imread(m_model_data->filename.substr(0,m_model_data->filename.find_last_of("."))+".skin_mask.png");
	cvtColor(tmp,*model_skin_mask,CV_BGR2GRAY);
}

Mat composeHead(Mat* _headMask, 
				Mat& skinMaskOutput,
				Mat* orig_im,
				Point* face_loc, Point* model_loc,
				VirtualSurgeon::VirtualSurgeonParams* m_params,
				VirtualSurgeon::VirtualSurgeonParams* m_model_data,
				double scaleFromFaceToBack) {	
	Mat background;
	Mat model_skin_mask;
	getModelImage(&background,&model_skin_mask,m_model_data);

	cout << endl << "---------------------- Recoloring --------------------" << endl;
	VirtualSurgeon::Recoloring recoloring(*m_params);
	Mat __headMask;
	if(_headMask->type() != CV_8U) {
		_headMask->convertTo(__headMask,CV_8UC1,255.0);
	} else
		__headMask = *_headMask;
	Mat* headMask = &__headMask;
	recoloring.Recolor(*orig_im,__headMask,background,model_skin_mask,skinMaskOutput);
	cvDestroyAllWindows();

	Rect faceRect;
	VirtualSurgeon::FindBoundingRect(faceRect,*headMask);

	cout << endl << "---------------------- Compose --------------------" << endl;
	//enlarge ROI by a bit
	faceRect.x = MAX(0,faceRect.x - 3*m_params->poisson_cloning_band_size);
	faceRect.y = MAX(0,faceRect.y - 3*m_params->poisson_cloning_band_size);
	faceRect.width = MIN(faceRect.width + 5*m_params->poisson_cloning_band_size,headMask->cols - faceRect.x);
	faceRect.height = MIN(faceRect.height + 5*m_params->poisson_cloning_band_size,headMask->rows - faceRect.y);

	cout << faceRect.x <<","<<faceRect.y<<","<<faceRect.width<<","<<faceRect.height << endl;

	//double model_dist = norm(m_model_data->li - m_model_data->ri);
	//model_dist /= cos(m_model_data->yaw / 180.0 * CV_PI);
	//double face_dist = norm(m_params->li - m_params->ri);
	//face_dist /= cos(m_params->yaw / 180.0 * CV_PI);

	//double scaleFromFaceToBack = model_dist / face_dist;

	//double scaleFromFaceToBack = norm(m_model_data->li - m_model_data->ri) / 
	//					norm(m_params->li - m_params->ri);
	//double scaleFromFaceToBack = 1.0;

 	Mat _tmp_o; 
	{
		(*orig_im)(faceRect).convertTo(_tmp_o,CV_32F);
		Mat __tmp;
		cout << "resize orig from " << _tmp_o.cols << "," << _tmp_o.rows << " to " << _tmp_o.cols * scaleFromFaceToBack << "," << _tmp_o.rows*scaleFromFaceToBack << "\n";
		if(m_params->blur_on_resize && scaleFromFaceToBack < 1.0) GaussianBlur(_tmp_o,_tmp_o,Size(5,5),1/sqrt(scaleFromFaceToBack));
		resize(_tmp_o,__tmp,Size(),scaleFromFaceToBack,scaleFromFaceToBack,CV_INTER_AREA);
		__tmp.copyTo(_tmp_o);
	}

	cout << "rsize mask\n";
	Mat mask; //(*headMask)(faceRect).copyTo(mask);
	//GaussianBlur(*headMask,*headMask,Size(5,5),1/scaleFromFaceToBack);
	resize((*headMask)(faceRect),mask,Size(),scaleFromFaceToBack,scaleFromFaceToBack);

	Point face_p(face_loc->x,face_loc->y);
	Point face_p1(	(int)((double)(face_loc->x-faceRect.x) * scaleFromFaceToBack), 
					(int)((double)(face_loc->y-faceRect.y) * scaleFromFaceToBack));

	Rect modelRect;
	modelRect.x = model_loc->x - face_p1.x;
	modelRect.y = model_loc->y - face_p1.y;
	//modelRect.x = face_p
	//modelRect.x += model_loc->x - face_p1.x;
	//modelRect.x += faceRect.width * scaleFromFaceToBack;
	//modelRect.y += model_loc->y - face_p1.y;
	//modelRect.y += faceRect.height * scaleFromFaceToBack;
	modelRect.width = _tmp_o.cols;
	modelRect.height = _tmp_o.rows;
	cout << modelRect.x <<","<<modelRect.y<<","<<modelRect.width<<","<<modelRect.height << endl;

	//if face area wants to go above model image space - cut the clipping rects
	if(modelRect.x < 0) { 
		cout << "modelRect overflow on x\n";
		faceRect.x -= modelRect.x; faceRect.width += modelRect.x; modelRect.x = 0;
		{
			(*orig_im)(faceRect).convertTo(_tmp_o,CV_32F);
			Mat __tmp;
			resize(_tmp_o,__tmp,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
			__tmp.copyTo(_tmp_o);
			resize((*headMask)(faceRect),mask,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
		}
		modelRect.width = _tmp_o.cols;
		modelRect.height = _tmp_o.rows;
	}
	if(modelRect.y < 0) { 
		cout << "modelRect overflow on y\n";
		faceRect.y -= modelRect.y; faceRect.height += modelRect.y; modelRect.y = 0; 
		{
			(*orig_im)(faceRect).convertTo(_tmp_o,CV_32F);
			Mat __tmp;
			resize(_tmp_o,__tmp,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
			__tmp.copyTo(_tmp_o);
			resize((*headMask)(faceRect),mask,Size(),scaleFromFaceToBack,scaleFromFaceToBack);
		}
		modelRect.width = _tmp_o.cols;
		modelRect.height = _tmp_o.rows;
	}

	/******* allow user to interact *******/
	//HeadPositionUserInteraction(*orig_im,*headMask,_tmp_o,mask,faceRect,background,modelRect,scaleFromFaceToBack);

	Mat _tmp_m; (background)(modelRect).convertTo(_tmp_m,CV_32F);


	//Eliminate any skin that penetrates the skin masks of the model
	{
		Mat modelMask_8UC; model_skin_mask(modelRect).convertTo(modelMask_8UC,CV_8UC1);
		//Mat modelSkinMask_8UC; (*model_skin_mask)(modelRect).convertTo(modelSkinMask_8UC,CV_8UC1);
		threshold(modelMask_8UC,modelMask_8UC,1.0,255.0,THRESH_BINARY);
		Mat mask_8UC; mask.convertTo(mask_8UC,CV_8UC1,255.0);
		threshold(mask_8UC,mask_8UC,1.0,255.0,THRESH_BINARY);

		if(!m_params->no_gui) {
			Mat tmp; cvtColor(modelMask_8UC,tmp,CV_GRAY2BGR);
			putText(tmp,"modelMask_8UC",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);

			cvtColor(mask_8UC,tmp,CV_GRAY2BGR);
			putText(tmp,"mask_8UC",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);
		}

		Mat A = modelMask_8UC & mask_8UC;
		if(!m_params->no_gui) {
			Mat tmp; cvtColor(A,tmp,CV_GRAY2BGR);
			putText(tmp,"A = modelMask_8UC & mask_8UC",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);
		}
		Mat B = mask_8UC - A;
		if(!m_params->no_gui) {
			Mat tmp; cvtColor(B,tmp,CV_GRAY2BGR);
			putText(tmp,"B = mask_8UC - A",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);
		}
		VirtualSurgeon::takeBiggestCC(B);
		if(!m_params->no_gui) {
			Mat tmp; cvtColor(B,tmp,CV_GRAY2BGR);
			putText(tmp,"takeBiggestCC(B)",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);
		}
		mask_8UC = B + A;
		if(!m_params->no_gui) {
			Mat tmp; cvtColor(mask_8UC,tmp,CV_GRAY2BGR);
			putText(tmp,"mask_8UC = B + A",Point(10,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2);
			imshow("tmp",tmp);
			waitKey(m_params->wait_time);
		}
			
		{
			Mat tmp; mask_8UC.convertTo(tmp,CV_32FC1,1.0/255.0);
			if(mask.type() == CV_8U) {
				Mat _tmp; mask.convertTo(_tmp,CV_32FC1,1.0/255.0); _tmp.copyTo(mask);
			}
			mask = mask.mul(tmp);
		}
	}

	Mat antiMask = -mask + 1.0;

	if(!m_params->no_gui) {
		namedWindow("tmp");
		imshow("tmp",mask);
		waitKey(m_params->wait_time);
		imshow("tmp",_tmp_m / 255.0);
		waitKey(m_params->wait_time);
	}

	vector<Mat> ov; split(_tmp_o,ov);
	vector<Mat> mv; split(_tmp_m,mv);
	for(int i=0;i<ov.size();i++) {
		ov[i] = ov[i].mul(mask) + mv[i].mul(antiMask);
	}
	Mat out;
	merge(ov,out);

	Mat modelMask; model_skin_mask(modelRect).convertTo(modelMask,CV_32FC1);
	{
		threshold(mask,mask,50.0/255.0,1.0,CV_THRESH_BINARY);

		//create a "band" around the contour

		//Dilate-Erode
		Mat dil; dilate(mask,dil,Mat::ones(m_params->poisson_cloning_band_size,m_params->poisson_cloning_band_size,CV_8UC1));
		Mat ero; erode (mask,ero,Mat::ones(m_params->poisson_cloning_band_size,m_params->poisson_cloning_band_size,CV_8UC1));
		mask = dil - ero;

		//Gaussian-Threshold
		GaussianBlur(mask,mask,Size(5,5),1.5);
		threshold(mask,mask,0.5,1.0,CV_THRESH_BINARY);
	}
	m_params->PoissonImageEditing(out,modelMask,_tmp_m,mask,true);

	Mat complete;

	background.copyTo(complete);
	Mat complete_modelRect = complete(modelRect);
	out.convertTo(complete_modelRect,CV_8UC3);

	if(!m_params->no_gui) {
		imshow("tmp",complete);
		waitKey(m_params->wait_time);
	}

	return complete;
}

void TrainUserLocationScale(VirtualSurgeon::VirtualSurgeonParams& face_params,
							const Mat& face_orig_image, 
							const Mat& face_orig_mask, 
							Rect& face_orig_im_offset,
							double& anchors_to_eyes_ratio_avg,
							Point2f& anchor_to_eye_norm_avg) {

	vector<string> dummies; vector<Mat_<Point2f> > dummies_anchors;
	if(face_params.is_female) {
//		dummies.push_back("D:\\Head Replacement\\VirtualSurgeon\\images\\dummy for location\\normal-female.png");
//		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(141,161),Point2f(226,161)));
//		dummies.push_back("D:\\Head Replacement\\VirtualSurgeon\\images\\dummy for location\\short-female.png");
//		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(183,158),Point2f(258,158)));
		dummies.push_back(face_params.path_to_exe + "3281508985_668495be2a_b_d.dummy.png");
		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(313,193),Point2f(372,190)));
	} else {
		dummies.push_back(face_params.path_to_exe + "3280223114_e244fca104_b_d.dummy.png");
		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(311,266),Point2f(366,266)));
//		dummies.push_back("D:\\Head Replacement\\VirtualSurgeon\\images\\dummy for location\\fat-dummy.png");
//		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(178,180),/* Point2f(196,189), Point2f(217,194), Point2f(236,197), Point2f(258,194), Point2f(276,188),*/ Point2f(295,178)));
//		dummies.push_back("D:\\Head Replacement\\VirtualSurgeon\\images\\dummy for location\\ShortSkinny-dummy.png");
//		dummies_anchors.push_back((Mat_<Point2f>(2,1) << Point2f(93,100),Point2f(152,100)));
	}

	anchors_to_eyes_ratio_avg = 0.0;
	anchor_to_eye_norm_avg = Point2f(0,0);

	for(int dummy_i=0;dummy_i<dummies.size();dummy_i++) {
		Mat _dummy_image = imread(dummies[dummy_i]);
		int pad = 160;
		Mat_<Vec3b> padding(_dummy_image.rows/2 + pad, _dummy_image.cols/2 + pad); padding.setTo(Scalar(255,255,255));
		Mat padding_pad = padding(Rect(pad/2,pad/2,_dummy_image.cols/2,_dummy_image.rows/2));
		resize(_dummy_image, padding_pad, Size(_dummy_image.cols/2,_dummy_image.rows/2));
		Mat dummy_image; padding.convertTo(dummy_image,CV_32F);

		Rect face_rect;
		VirtualSurgeon::FindBoundingRect(face_rect,face_orig_mask);
		Mat face_area_image = face_orig_image(face_rect);
		Mat face_mask = face_orig_mask(face_rect);
		Rect dummy_rect(100,100,face_rect.width,face_rect.height);
		double scaleFromFaceToBack = 1.0;

		HeadPositionUserInteraction(
			face_orig_image,
			face_orig_mask,
			face_area_image,
			face_mask,
			face_rect,
			dummy_image,
			dummy_rect,
			scaleFromFaceToBack);

		Mat_<Point2f> anchors = dummies_anchors[dummy_i];
		anchors = anchors/2.0 + Scalar(pad/2,pad/2);

		Point2f face_params_li_f = face_params.li;
		Point2f face_params_ri_f = face_params.ri;
		Mat_<Point2f> eyes = (Mat_<Point2f>(2,1) << face_params_li_f, face_params_ri_f);
		eyes = (eyes) * scaleFromFaceToBack + Scalar(dummy_rect.x-face_rect.x * scaleFromFaceToBack,dummy_rect.y-face_rect.y * scaleFromFaceToBack);

		//Mat _tmp; Mat(dummy_image/255.0).copyTo(_tmp);
		//for(int i=0;i<anchors.rows;i++) {
		//	circle(_tmp,anchors(i),3,Scalar(0,0,255),3);
		//	line(_tmp,anchors(i),eyes(0),Scalar(0,255,255),2);
		//	line(_tmp,anchors(i),eyes(1),Scalar(255,0,255),2);
		//}
		//rectangle(_tmp,dummy_rect,Scalar(255));
		//rectangle(_tmp,Rect(dummy_rect.x-face_rect.x * scaleFromFaceToBack,dummy_rect.y-face_rect.y * scaleFromFaceToBack,face_orig_image.cols * scaleFromFaceToBack,face_orig_image.rows * scaleFromFaceToBack),Scalar(0,255));
		//imshow("_tmp",_tmp);
		//waitKey();

		double a_d = norm(anchors(0) - anchors(1));
		double e_d = norm(eyes(0) - eyes(1));

		double anchors_to_eyes_ratio = e_d/a_d;
		Point2f anchor_to_eye_normalized = (anchors(0) - eyes(0)) * (1.0/a_d);

		cout << "anchors_d " << a_d <<", eye_d " << e_d << ", anchors_to_eyes_ratio " << anchors_to_eyes_ratio << ", anchor_to_eye_normalized " << anchor_to_eye_normalized << endl;

		anchors_to_eyes_ratio_avg += anchors_to_eyes_ratio;
		anchor_to_eye_norm_avg += anchor_to_eye_normalized;

		//Mat_<Vec4f> transformations(1,1);
		//int indxs[2] = {0,1} ;//,	0,2,	1,2};
		//Point2f ve = eyes(0)-eyes(1);
		////for(int i=0;i<1;i++) 
		//int i=0;
		//{
		//	Point2f va = anchors(indxs[2*i])-anchors(indxs[2*i+1]);
		//	transformations(i).val[0] = acos((ve*(1.0/norm(ve))).dot(va*(1.0/norm(va)))); //angle
		//	Point2f d = (anchors(indxs[2*i]) - eyes(0))  * (1.0 / norm(va));
		//	transformations(i).val[1] = d.x; //tx
		//	transformations(i).val[2] = d.y; //ty
		//	transformations(i).val[3] = norm(va)/norm(ve); //scale
		//}

		//Mat(dummy_image/255.0).copyTo(_tmp);
		////for(int i=0;i<2;i++) 
		////int i=0;
		//{
		//	Point2f va = anchors(indxs[2*i])-anchors(indxs[2*i+1]);
		//	float a = transformations(i).val[0];
		//	va = Point2f(va.x * cos(a) + va.y * -sin(a), va.x * sin(a) + va.y * cos(a)); //rotate
		//	va = va * (1.0f/transformations(i).val[3]); //magnify
		//	Point2f estimated_eye1 = anchors(indxs[2*i]) - (Point2f(transformations(i).val[1],transformations(i).val[2]) * norm(va));
		//	Point2f estimated_eye2 = estimated_eye1 - va; //translate

		//	circle(_tmp,estimated_eye1,6,Scalar(255),1);
		//	circle(_tmp,estimated_eye2,6,Scalar(0,255),1);
		//}
		//circle(_tmp,eyes(0),2,Scalar(255),2);
		//circle(_tmp,eyes(1),2,Scalar(0,255),2);

		//imshow("_tmp",_tmp);
		//waitKey();
	}

	anchors_to_eyes_ratio_avg /= dummies.size();
	anchor_to_eye_norm_avg *= (1.0/dummies.size());

	cout << "average anchors_to_eyes_ratio " << anchors_to_eyes_ratio_avg << ", average anchor_to_eye_normalized " << anchor_to_eye_norm_avg << endl;
}

void StartLandmarksWindow(const cv::Mat& im, VirtualSurgeon::VirtualSurgeonParams& p);
void StartMainWindow( VirtualSurgeon::VirtualSurgeonParams& p);

int head_extractor_main(int argc, char** argv) {
	cout << "Head extractor" << endl;
	
	
	try {
		VirtualSurgeon::VirtualSurgeonParams p;
		p.InitializeDefault();
		const char* _argv[] = {
			"dummy",
			"--groundtruth 2-Normal.png",
			"-r 1",
			"-b 1",
			"--gabor-size 0",
			"-x 3",
			"--num-DoG 0",
			"-k",
			"-c 20",
			"--poisson-blend-band 8",
			"-u",
			"--do-two-back-kernels",
			"--no-gui",
			"--wait-time 1"};
		
		p.ParseParams(14,_argv);
		
		StartMainWindow(p);
		
		/*	p.im_scale_by = 1.3;
		 p.no_gui = false;
		 p.wait_time = 0;
		 p.gc_iter = 0;
		 p.num_DoG = 0;
		 p.gb_size = 8;
		 p.km_numc = 40;
		 p.hair_ellipse_size_mult = 1.2;
		 p.do_alpha_matt = false;
		 p.consider_pixel_neighbourhood = true;
		 p.do_two_segments = false;
		 p.do_kmeans = true;
		 p.do_two_back_kernels = true;
		 p.doPositionInKM = false;
		 */
		
		
		if (p.filename == "") {
			//TODO use fltk3 instead
//			QApplication app(argc,argv);
//			p.filename = QFileDialog::getOpenFileName(0, "Open Image", QString(), QString("Image Files (*.png *.jpg *.bmp)")).toStdString();
			p.filename = string(fltk3::file_chooser("Open a file","Image Files (*.{bmp,gif,jpg,png})","",0));
		}
		
		//p.filename = std::string(argv[1]);
		Mat _tmp,im;
		if(!p.FaceDotComDetection(_tmp)) {
				//no cached information
			//detect eyes
#ifdef VIRTUAL_SURGEON_DEBUG
			fltk3::alert("Detect Eyes");
#endif
			p.DetectEyes(_tmp);
		}
		
		/***************** aid with landmarks positioning *******************/
#ifdef VIRTUAL_SURGEON_DEBUG
		fltk3::alert("Landmarks");
#endif
		StartLandmarksWindow(_tmp,p);
		
		p.PrintParams();
		//resize(_tmp,im,Size(),0.75,0.75);
		//p.li *= 0.75;
		//p.ri *= 0.75;
		
		_tmp.copyTo(im);
		
		p.pitch /= 5.0;
		p.yaw /= 5.0;
		//_tmp.copyTo(im);
		
		//Cropping a big enough rectange around the face
		double li_ri = norm(p.li - p.ri);// / (double)(faceMask.cols);
		int li_ri_m_3 = (int)(li_ri*3.5);
		int li_ri_t_6_5 = (int)(li_ri*7.5);
		Rect r(MIN(im.cols,MAX(0,p.li.x - li_ri_m_3)),
			   MIN(im.rows,MAX(0,p.li.y - li_ri_m_3)),
			   MIN(im.cols-MAX(0,p.li.x - li_ri_m_3),MAX(0,li_ri_t_6_5)),
			   MIN(im.rows-MAX(0,p.li.y - li_ri_m_3),MAX(0,li_ri_t_6_5)));
		r = r & Rect(0,0,im.cols,im.rows);
		
		im(r).copyTo(_tmp);
		
		//set the eyes to match the new ROI
		Point orig_li = p.li;
		Point orig_ri = p.ri;
		p.li = p.li - r.tl(); 
		p.ri = p.ri - r.tl(); 
		
#ifdef VIRTUAL_SURGEON_DEBUG
		fltk3::alert("Face relighting");
#endif
		cout << endl << "--------------------- Face relighting ----------------------" << endl;
#if 1
		{
			Mat _relitFace,_relitMask; Rect relitRect;
			int spharmonics_res = SpharmonicsUI_main(argc, argv, p, _tmp, _relitFace, _relitMask, relitRect);
			if (!spharmonics_res) { //no error occured in relighting
				int padding = 20;
				relitRect.x -= padding;
				relitRect.y -= padding;
				relitRect.width += padding*2;
				relitRect.height += padding*2;
				
				Mat roi;
				
				Mat relitFace(relitRect.size(),_relitFace.type(),Scalar(0));
				roi = relitFace(Rect(padding,padding,_relitFace.cols,_relitFace.rows));
				_relitFace.copyTo(roi);
				
				Mat relitMask(relitRect.size(),_relitMask.type(),Scalar(0));
				roi = relitMask(Rect(padding,padding,_relitMask.cols,_relitMask.rows));
				_relitMask.copyTo(roi);
				
				if(!p.no_gui) {		  
					Mat_<Vec3b> all(relitRect.height,relitRect.width*4);
					roi = all(Rect(0,0,relitRect.width,relitRect.height));
					_tmp(relitRect).convertTo(roi, CV_8UC3);	
					roi = all(Rect(relitRect.width,0,relitRect.width,relitRect.height));
					cvtColor(~relitMask, roi, CV_GRAY2BGR);	
					roi = all(Rect(relitRect.width*2,0,relitRect.width,relitRect.height));
					relitFace.convertTo(roi, CV_8UC3, 255.0);
					roi = all(Rect(relitRect.width*3,0,relitRect.width,relitRect.height));
					cvtColor(relitMask, roi, CV_GRAY2BGR);	
					imshow("all",all); waitKey(p.wait_time);
					cv::destroyAllWindows();
				}
				
				//Use poisson belnding to create a "continuation" of the relit face, for blending into original
				Mat relitFace8UC3; relitFace.convertTo(relitFace8UC3,CV_8UC3,255.0);
				bool no_gui = p.no_gui; int wait_time = p.wait_time;
				p.no_gui = true; p.wait_time = 1;
				p.PoissonImageEditing(relitFace8UC3, 
									  Mat(~relitMask), 
									  _tmp(relitRect), 
									  Mat_<float>::ones(relitMask.size()),  
									  true, 
									  false);
				p.no_gui = no_gui; p.wait_time = wait_time;
				if(!p.no_gui) {
					imshow("tmp",relitFace8UC3); waitKey(p.wait_time);
				}
				
				//Blend the relit face into the original head
				{
					Mat_<Vec3f> relitFace32F3; relitFace8UC3.convertTo(relitFace32F3, CV_32FC3, 1.0/255.0);			
					Mat_<Vec3f> orig32F3; _tmp(relitRect).convertTo(orig32F3, CV_32FC3, 1.0/255.0);			
					Mat_<float> blendMask; Mat(relitMask).convertTo(blendMask,CV_32FC1, 1.0/255.0);
					
					if(!p.no_gui) {		  
						Mat_<Vec3f> all(relitRect.height,relitRect.width*3);
						roi = all(Rect(0,0,relitRect.width,relitRect.height));
						relitFace32F3.copyTo(roi);
						roi = all(Rect(relitRect.width,0,relitRect.width,relitRect.height));
						cvtColor(blendMask, roi, CV_GRAY2BGR);	
						roi = all(Rect(relitRect.width*2,0,relitRect.width,relitRect.height));
						orig32F3.copyTo(roi);
						imshow("all",all); waitKey(p.wait_time);
						destroyAllWindows();
					}			
					
					Mat_<Vec3f> res = LaplacianBlend(relitFace32F3,orig32F3,blendMask);
					if(!p.no_gui) {
						imshow("orig",orig32F3);
						imshow("tmp",res); waitKey(p.wait_time);
						destroyAllWindows();
					}
					
					roi = _tmp(relitRect);
					res.convertTo(roi, CV_8UC3, 255.0);
				}
			}
		}
#endif		
		VirtualSurgeon::HeadExtractor he(p);
		cout << endl << "---------------------- Extract Head --------------------" << endl;
		Mat skinMaskOutput;
		Mat maskFace = he.ExtractHead(_tmp,r,&skinMaskOutput);
		//	imshow("result",result);
		/*	if (!p.output_filename.empty()) {
		 vector<Mat> v;
		 Mat imMasked,_result;
		 im(r).convertTo(imMasked,CV_32FC3,1.0/255.0);
		 split(imMasked,v);
		 v[0] = v[0].mul(maskFace) + Mat::ones(maskFace.size(),CV_32FC1) - maskFace;
		 v[1] = v[1].mul(maskFace);
		 v[2] = v[2].mul(maskFace);
		 cv::merge(v,imMasked);
		 imMasked.convertTo(_result,CV_8UC3,255.0);
		 
		 cout << "write: " << p.output_filename << "\n";
		 
		 //imshow("result",_result);
		 imwrite(p.output_filename,_result);
		 
		 string s = p.output_filename + "_orig.png";
		 cout << "write: " << s << "\n";
		 imwrite(s,im(r));
		 }
		 */
		
		{
			Mat _roi = im(r);
			if(_tmp.type() == im.type() && _tmp.size() == r.size()) {
				_tmp.copyTo(_roi);
			} else if(_tmp.type() == im.type() && _tmp.size() != r.size()) {
				resize(_tmp,_roi,r.size());
			} else if (_tmp.type() != im.type() && _tmp.size() == r.size()) {
				_tmp.convertTo(_roi, im.type(), (im.type() == CV_8UC3)?255.0:1.0/255.0);
			}

			if(!p.no_gui) {
				imshow("_tmp",_tmp);
				imshow("im(r)",im(r));
				waitKey(p.wait_time);
			}
		}
		
		Mat_<Point2f>& curve = p.LoadHeadCurve(im,r);
		//Point2f pp = *(curve[134])*0.4 + (*(curve[66]) + *(curve[62])) * 0.3;
		//Point2f pp = curve(30)*0.4 + (curve(25) + curve(35)) * 0.3;
		Point2f pp = curve(134)*0.6 + (curve(25) + curve(35)) * 0.2;
		//{
		//	Mat _curve_out; im(r).copyTo(_curve_out);
		//	for(int i=0;i<135;i++) {
		//		circle(_curve_out,*(curve[i]),3,Scalar(0,255,0),CV_FILLED);
		//	}
		//	circle(_curve_out,pp,3,Scalar(255,0,255));
		//	line(_curve_out,pp,curve(30),Scalar(255,255,0),1);
		//	line(_curve_out,pp,curve(25),Scalar(255,255,0),1);
		//	line(_curve_out,pp,curve(35),Scalar(255,255,0),1);
		//	imshow("tmp1",_curve_out); waitKey();
		//}
		

		
		Point model_loc;
		Point face_loc(pp.x,pp.y);
		
		cout << endl << "---------------------- Get user training in transplating heads --------------------" << endl;
		
		double anchors_to_eyes_ratio_avg;
		Point2f anchor_to_eye_norm_avg;
		TrainUserLocationScale(p,im(r),maskFace,r,anchors_to_eyes_ratio_avg,anchor_to_eye_norm_avg);
		
		
		
		vector<string> models;
		//after the "-" in p.groundtruth is the 'body size': "Normal","Fat",etc.
		string body_size = p.groundtruth.substr(p.groundtruth.find_last_of("-"));
		if(p.is_female) {
			models.push_back(p.path_to_exe + "reshaping/4" + body_size);
			models.push_back(p.path_to_exe + "reshaping/5" + body_size);
			models.push_back(p.path_to_exe + "reshaping/6" + body_size);
			models.push_back(p.path_to_exe + "reshaping1/1" + body_size);
			models.push_back(p.path_to_exe + "reshaping1/4" + body_size);
		} else {
			models.push_back(p.path_to_exe + "reshaping/1" + body_size);
			models.push_back(p.path_to_exe + "reshaping/2" + body_size);
			models.push_back(p.path_to_exe + "reshaping/3" + body_size);
			models.push_back(p.path_to_exe + "reshaping1/2" + body_size);
			models.push_back(p.path_to_exe + "reshaping1/3" + body_size);
		}
		
		for(int i=0;i<models.size();i++) {
			VirtualSurgeon::VirtualSurgeonParams model_params; 
			model_params.InitializeDefault();
			model_params.filename = models[i]; //p.groundtruth;
			
			char _tmpcptr[10];
			{
				stringstream ss; ss << p.filename.substr(p.filename.find_last_of("\\")+1,p.filename.find_last_of("\\")+1-p.filename.find_last_of(".")) << "-" << i << ".png";
				p.output_filename = ss.str();	
			}
			
			{
				string infoFile = model_params.filename.substr(0,model_params.filename.find_last_of(".")) + ".model_info.txt";
				ifstream ifs(infoFile.c_str(),ifstream::in);
				if(ifs.is_open()) {
					char delim;
					ifs >> skipws
					>> model_loc.x //>> delim 
					>> model_loc.y //>> delim 
					>> model_params.li.x //>> delim 
					>> model_params.li.y
					>> model_params.ri.x //>> delim 
					>> model_params.ri.y;
				} else {
					cerr << "cannot read model_info file: " << infoFile << endl;
					continue;
				}
			}
			
			model_params.yaw = 0.0;
			
			double anchors_d = norm(model_params.li - model_params.ri);
			double eyes_d = norm(p.li - p.ri);
			double scaleFromFaceToBack = (anchors_to_eyes_ratio_avg * anchors_d / eyes_d);
			face_loc = p.li;
			model_loc.x = model_params.li.x - anchor_to_eye_norm_avg.x * (float)anchors_d;
			model_loc.y = model_params.li.y - anchor_to_eye_norm_avg.y * (float)anchors_d;
			
			
			cout << endl << "---------------------- Compose Head --------------------" << endl;
			Mat composed = composeHead(&maskFace,skinMaskOutput,&(im(r)),&face_loc,&model_loc,&p,&model_params,scaleFromFaceToBack);
						
			//Scale down to match input image resolution...
			Mat composed_small; resize(composed, composed_small, Size(), 1.0/scaleFromFaceToBack,1.0/scaleFromFaceToBack);
			
			cout << "write: " << p.output_filename << endl;
			if(!p.output_filename.empty()) imwrite(p.output_filename,composed_small);
			
			if(!p.no_gui) waitKey();
			
			imshow(p.output_filename,composed_small);
			waitKey(200);
		}

		cv::destroyAllWindows();
		
		fltk3::input("Thank you for using the application, your unique key: ",p.GenerateUniqueID().c_str());

	} catch (Exception e) {
		cerr << "Error " << e.what();
		fltk3::alert(e.what());
		waitKey();
		return 1;
	}
	
	

	return 0;
	/*
	 Mat mask1(_tmp.size(),CV_8UC1,Scalar(0)),mask2(_tmp.size(),CV_8UC1,Scalar(0));
	 he.CreateEllipses(_tmp,mask1,mask2,Point());
	 Mat maskfinal = mask1 | mask2;
	 
	 im(r).copyTo(_tmp);
	 namedWindow("tmp");
	 namedWindow("tmp1");
	 p.face_grab_cut(_tmp,maskfinal,1);
	 
	 vector<Mat> v; split(im(r),v);
	 //v[0] = Mat::zeros(maskfinal.size(),CV_8UC1);
	 v[1] = v[1] * 0.7 + maskfinal * 0.3;
	 v[2] = v[2] * 0.7 + ~maskfinal * 0.3;// + 255.0;
	 cv::merge(v,_tmp);
	 imshow("tmp",_tmp);
	 _tmp.setTo(Scalar(0));
	 im(r).copyTo(_tmp,maskfinal);
	 imshow("tmp1",_tmp);
	 waitKey();
	 
	 {
	 Mat tmpMask(maskfinal.rows,maskfinal.cols,CV_8UC1,Scalar(0));
	 int dilate_size = 5; //p.alpha_matt_dilate_size;
	 
	 //prepare trimap
	 {
	 Mat __tmp(maskfinal.rows,maskfinal.cols,CV_8UC1,Scalar(0));
	 dilate(maskfinal,__tmp,Mat::ones(dilate_size,dilate_size,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT);	//probably background
	 tmpMask.setTo(Scalar(128),__tmp);
	 
	 erode(maskfinal,__tmp,Mat::ones((int)((double)(dilate_size)*1.5),
	 (int)((double)(dilate_size)*1.5),CV_8UC1),Point(-1,-1),1,BORDER_REFLECT); // foreground
	 tmpMask.setTo(Scalar(255),__tmp);
	 }
	 
	 Mat tmpim; im(r).copyTo(tmpim);
	 
	 imshow("tmp",tmpMask);
	 imshow("tmp1",tmpim);
	 waitKey();
	 
	 Matting *matting = new BayesianMatting( &((IplImage)tmpim), &((IplImage)tmpMask) );
	 //Matting *matting = new RobustMatting( &((IplImage)im), &((IplImage)tmpMask) );
	 matting->Solve(!p.no_gui);
	 
	 Mat(matting->alphamap).copyTo(maskfinal);
	 
	 //maskFace.convertTo(maskFace,CV_8UC1,255);
	 
	 delete matting;
	 }
	 Mat unMask = Mat::ones(maskfinal.size(),CV_32FC1) - maskfinal; //Mat(maskFace.size(),CV_8UC1,Scalar(255)) - maskFace;
	 vector<Mat> v1;
	 Mat im1;
	 im(r).convertTo(im1,CV_32FC3,1.0/255.0);
	 split(im1,v1);
	 v1[0] = v1[0].mul(maskfinal) + (unMask);
	 v1[1] = v1[1].mul(maskfinal);
	 v1[2] = v1[2].mul(maskfinal);
	 Mat imMasked;
	 cv::merge(v1,imMasked);
	 
	 imshow("tmp1",imMasked);
	 waitKey();
	 */
}
#endif