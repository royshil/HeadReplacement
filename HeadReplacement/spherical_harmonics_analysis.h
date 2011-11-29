/*
 *  spherical_harmonics_analysis.h
 *  FaceTracker
 *
 *  Created by Roy Shilkrot on 10/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#import <opencv2/opencv.hpp>
#import <map>
#import <set>

using namespace cv;
using namespace std;

class SphericalHarmonicsAnalyzer {
private:
	//	vector<Point2d>& face_points; 
	
	Mat calculateSphericalHarmonicsForNormal(Vec3f n) {
		Mat_<float> sph(1,9);
		float nx = n.val[0], ny = n.val[1], nz = n.val[2];
		Vec3f nsq = n.mul(n);
		sph(0) = 0.282094792;	//................................  1.0 / sqrt(4 * pi);
		sph(1) = 1.02332671 * nz;	// ...........................  ((2 * pi) / 3) * sqrt(3 / (4 * pi))
		sph(2) = 1.02332671 * ny;
		sph(3) = 1.02332671 * nx;
		sph(4) = 0.247707956 * (2.0*nsq[2] - nsq[0] - nsq[1]);	//	(pi / 4) * (1 / 2) * sqrt(5 / (4 * pi))
		sph(5) = 0.858085531 * ny * nz;	//........................	(pi / 4) * 3 * sqrt(5 / (12 * pi))
		sph(6) = 0.858085531 * nx * nz;
		sph(7) = 0.858085531 * nx * ny;
		sph(8) = 0.429042765 * (nsq[0] - nsq[1]); //..............	(pi / 4) * (3 / 2) * sqrt(5 / (12 * pi))
		return sph;
	}

	vector<Vec3f> face_points_normals; 
	map<int,Vec3f> face_point_to_normal;
	map<int,Vec3f> face_point_to_vertex;
	Mat_<float> l; //lighting coefficients
	Mat face_img;
	Mat_<uchar> face_mask;
	vector<Point2d> face_points;
	set<Point2d> face_points_set;
	Mat_<Vec3f> normalMap;
	Mat_<Vec3f> albedo;
	
	double scaleFactor;
	
public:
	bool _debug;
	
	SphericalHarmonicsAnalyzer(const Mat_<Vec3b>& _face_img,
							   const Mat_<uchar>& _face_mask,
							   const Mat_<Vec3f>& _face_normals):
	face_img(_face_img),
	face_mask(_face_mask),
	normalMap(_face_normals), 
	_debug(true),
	scaleFactor(0.66)
	{};
	
	SphericalHarmonicsAnalyzer(map<int,Vec3f>& _ptn, /* model normals */
							   map<int,Vec3f>& _ptv, /* model vertices */
							   const Mat& _face_img, /* sample image */
							   Mat& _face_mask,		 /* sample mask */
							   vector<Point2d>& _face_points /* sample points */
							   ):
		face_point_to_normal(_ptn) ,
		face_point_to_vertex(_ptv),
		face_img(_face_img),
		face_mask(_face_mask),
		face_points(_face_points)
	{
		for (int i=0; i<face_points.size(); i++) { 
			Vec3f nv = _ptn[i];
			nv = nv * (1.0f / norm(nv));
			cout << face_points[i] << ": " << nv[0]<<","<<nv[1]<<","<<nv[2] << endl;
			face_points_normals.push_back(nv);
		}
//		face_points_set = set<Point2d>(_face_points.begin(),_face_points.end(),std::equal<Point2d>);
		normalMap = Mat_<Vec3f>(face_img.size()); //3D image
	};
	
	const Mat_<Vec3f>& getAlbedo() { return albedo; }
	const Mat_<uchar>& getMask() { return face_mask; }
	
	const Mat_<float>& getLightingCoefficients() { return l; }
	
	void renderWithCoefficients(const Mat_<float>& _l) {
		Mat l_save; l.copyTo(l_save);
		_l.copyTo(l);
		computeAlbedo();
		l_save.copyTo(l);
	}
	
	void align2Dto3D(Mat_<double>& Qx,Mat_<double>& Qy,Mat_<double>& Qz, Mat_<double>& tvec) {
		vector<Vec2f> image_points;
		vector<Vec3f> model_points;
		for (int i=0; i<face_points.size(); i++) {
			map<int,Vec3f>::iterator itr = face_point_to_vertex.find(i);
			if (itr != face_point_to_vertex.end()) {
				Vec2f v; v[0] = face_points[i].x; v[1] = face_points[i].y;
				image_points.push_back(v);
				model_points.push_back((*itr).second);
				cout << (Point2f)(image_points.back()) << "->" << (Point3f)(model_points.back()) << endl;
			}
		}
		vector<double> rvec(3);
		Mat_<double> camMatrix = (Mat_<double>(3,3) <<	1 , 0.0 , face_img.cols/2.0 ,
															0.0 , 1 , face_img.rows/2.0 ,
														0.0 , 0.0 , 1.0 );
		solvePnP(model_points, image_points, camMatrix, Mat_<double>(1,4), rvec, tvec, false);
		
		Mat_<double> rotMat; Rodrigues(rvec, rotMat);
		Mat_<double> R,Q;
		RQDecomp3x3(rotMat, R, Q, Qx, Qy, Qz);
		cout << "rotation around x " <<endl<< Qx <<endl<< "rotation around y " <<endl<< Qy<<endl<<"rotation around z "<<endl<<Qz<<endl;
		cout << "translation " << tvec <<endl;
	}
	
	void setDenseNormalMap(Mat_<Vec3f> _map) {
		_map.copyTo(normalMap);
	}
	
	void approximateDenseNormalMap() {
		//TODO: openMP!
		for (int y=0; y<face_img.rows; y++) {
			for (int x=0; x<face_img.cols; x++) {
				if (face_mask(y,x) == 0) {
					normalMap(y,x) = Vec3f(0,0,0);
					continue;
				}
//				vector<Point2d>::iterator pos;
//				if((pos = std::find(face_points.begin(),face_points.end(),Point2d(x,y))) != face_points.end()) {
//					normalMap(y,x) = face_point_to_normal[];
//				}
				//prepare weights
//				Mat_<Vec2d> p = (Mat_<Vec2d>(1,1) << Vec2d(x , y));
//				Mat_<Vec2d> _ps; repeat(p, face_points.size(), 1, _ps);
//				Mat face_points_mat(face_points);
//				Mat_<Vec2d> _fps = face_points_mat; //.reshape(1); //one channel..
//				Mat_<Vec2d> D = _fps-_ps;
//				D = D.reshape(face_points.size(),1);
//				Mat W; normalize(D,W);
				vector<float> _W(face_points.size());
				int exactPoint = -1;
				for (int i=0; i<face_points.size(); i++) {
					Vec2d d = Point2d(x,y)-face_points[i];
					float nd = (float)norm(d);
					if (fabsf(nd) < 0.000001f) {
						exactPoint = i;
						break;
					}
					float _d = 1.0f / (nd*nd);
					_W[i] = _d; //,_d,_d);
				}
				if (exactPoint >= 0) {
					normalMap(y,x) = face_points_normals[exactPoint];
					continue;
				}
				Scalar _sum = sum(_W);
				Mat W(_W); W = W * (1.0f / (float)_sum[0]);
				
				Mat _normal = Mat(face_points_normals).reshape(1).t() * W; //.reshape(1);
				Vec3f nv = _normal.at<Vec3f>(0);
				float sc = 1.0/norm(nv);
				normalMap(y,x) = nv * sc;
//				cout << normalMap(y,x)[0] << "," << normalMap(y,x)[1] << "," << normalMap(y,x)[2] << endl;
			}
		}
		imshow("normal map",(normalMap * 0.5f) + Scalar(0.5,0.5,0.5));
		if(_debug) waitKey(0);
	}
	
	void approximateInitialLightingCoeffs() {
		double t = getTickCount();
		Size small(round(face_img.cols*scaleFactor),round(face_img.rows*scaleFactor));
		int flatSize = small.width*small.height;
		
		Mat_<Vec3f> smallNormalMap; resize(normalMap,smallNormalMap,small);
		Mat_<Vec3f> normalMapFlat = smallNormalMap.reshape(flatSize);
		
		Mat_<Vec3b> smallFaceImage; resize(face_img,smallFaceImage,small);
		Mat_<Vec3b> face_img_hsv; cvtColor(smallFaceImage, face_img_hsv, CV_BGR2HSV);
		face_img_hsv = face_img_hsv.reshape(flatSize);
		
		Mat_<uchar> smallFaceMask; resize(face_mask,smallFaceMask,small,0,0,INTER_NEAREST);
		smallFaceMask = smallFaceMask.reshape(flatSize);

		int n = countNonZero(smallFaceMask);

		//average face value as albedo estimate
		Scalar albedo_constant = mean(face_img_hsv, smallFaceMask);
		
		//setup linear equation system, lighting coefficients (l) is unknown
		//I = p00 * Ht * l
		float p00 = (float)albedo_constant[2] / 255.0f;
		
		cout << "Build Ht("<<n<<",9)...";
		cout << "Build I("<<n<<",1)...";
		//build Ht and I
		Mat_<float> Ht(n,9);
		Mat_<float> I(n,1);
		int pos = 0;
		vector<Mat_<uchar> > face_img_chnls; split(face_img_hsv, face_img_chnls);
//		#pragma omp parallel for schedule(dynamic)
		for (int i=0; i<normalMapFlat.rows; i++) {
			if (smallFaceMask(i) == 0) {
				continue;
			}
			Ht.row(pos) = p00 * calculateSphericalHarmonicsForNormal(normalMapFlat(i));
			I(pos,0) = face_img_chnls[2](i) / 255.0f; //get V from HSV of pixel [0,1]
			pos ++;
		}
		cout << "DONE"  << endl;
				
		cout << "Solve" <<endl;
		solve(Ht, I, l, DECOMP_SVD);
		
		cout << "initial lighting coeffs: ";
		for (int i=0; i<l.rows; i++) {
			cout<<l.at<float>(i)<<",";
		}
		cout << endl;
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "approximateInitialLightingCoeffs: " << t <<"s"<< endl;
	}
	
	void computeLightingCoefficients() {
		Size small(round(face_img.cols*scaleFactor),round(face_img.rows*scaleFactor));
		int flatSize = small.width*small.height;
		
		Mat_<Vec3f> smallNormalMap; resize(normalMap,smallNormalMap,small);
		Mat_<Vec3f> normalMapFlat = smallNormalMap.reshape(flatSize);
		
		vector<Mat_<uchar> > face_img_chnls;
		{
			Mat_<Vec3b> smallFaceImage; resize(face_img,smallFaceImage,small);
			Mat_<Vec3b> face_img_hsv; cvtColor(smallFaceImage, face_img_hsv, CV_BGR2HSV);
			face_img_hsv = face_img_hsv.reshape(flatSize);
			split(face_img_hsv, face_img_chnls);
		}
		
		Mat_<uchar> smallFaceMask; resize(face_mask,smallFaceMask,small,0,0,INTER_NEAREST);
		smallFaceMask = smallFaceMask.reshape(flatSize);

		Mat_<float> grayAlbedo; cvtColor(albedo, grayAlbedo, CV_BGR2GRAY);
		Mat_<float> smallAlbedo; resize(grayAlbedo,smallAlbedo,small);
		smallAlbedo = smallAlbedo.reshape(flatSize);
		
		int n = countNonZero(smallFaceMask);
		
		//setup linear equation system, lighting coefficients (l) is unknown
		//I = p00 * Ht * l
		
		cout << "Build Ht("<<n<<",9)...";
		cout << "Build I("<<n<<",1)...";
		//build Ht and I
		Mat_<float> Ht(n,9);
		Mat_<float> I(n,1);
		int pos = 0;
		#pragma omp parallel for schedule(dynamic)
		for (int i=0; i<normalMapFlat.rows; i++) {
			if (smallFaceMask(i) == 0) {
				continue;
			}
			Ht.row(pos) = smallAlbedo(i) * calculateSphericalHarmonicsForNormal(normalMapFlat(i));
			I(pos,0) = face_img_chnls[2](i) / 255.0f; //get V from HSV of pixel [0,1]
			pos ++;
		}
		cout << "DONE"  << endl;
		
		cout << "Solve" <<endl;
		solve(Ht, I, l, DECOMP_SVD);
		
		cout << "lighting coeffs: ";
		for (int i=0; i<l.rows; i++) {
			cout<<l.at<float>(i)<<",";
		}
		cout << endl;
	}
	
	void computeAlbedo() {
		double t = getTickCount();
		if (albedo.data == 0) {
			albedo.create(face_img.size());
		}
		if(l(0)!=l(0)) { cerr << "lighting coeffs are nan." << endl; return; }
		
		Mat_<Vec3f> _albedo_and_original(albedo.rows,albedo.cols*2+face_img.cols);
		_albedo_and_original.setTo(Scalar(0,0,0));

		
//		Mat_<float> face_img_v; 
//		Mat_<Vec3b> face_img_hsv; cvtColor(face_img, face_img_hsv, CV_BGR2HSV);
//		//{	//convert V (of HSV) to [0,1] range
//			vector<Mat_<uchar> > chnls; split(face_img_hsv, chnls);
////			chnls[2].convertTo(face_img_v,CV_32F,1.0/255.0);
//		//}
		
		Mat_<Vec3b> face_img_v3b = face_img;
		
		#pragma omp parallel for schedule(dynamic)
		for (int y=0; y<face_img.rows; y++) {
			for (int x=0; x<face_img.cols; x++) {
				if (face_mask(y,x) == 0) {
					albedo(y,x) = 0;
					continue;
				}
				Mat sph = calculateSphericalHarmonicsForNormal(normalMap(y,x));
//				vector<float> sphf; sph.copyTo(sphf);
				Mat_<float> sph_l = sph * l;
				float fsph_l = sph_l(0);

				for (int cn = 0; cn<3; cn++) {					
					float fimg = face_img_v3b(y,x)[cn] / 255.0f;
					albedo(y,x)[cn] = (fimg / fsph_l);
				}
			}
		}

		Mat roi;
//		Mat roi = _albedo_and_original(Rect(Point(albedo.cols+face_img_hsv.cols,0),face_img_hsv.size()));
//		face_img_hsv.convertTo(roi, CV_32F, 1.0/255.0);
//		face_img.convertTo(roi, CV_32F, 1.0/255.0);
				
//		cvtColor(albedo, roi, CV_GRAY2BGR);
#if 0
		{
			Mat albedo_8uc3; albedo.convertTo(albedo_8uc3,CV_8UC3, 255.0);
			cvtColor(albedo_8uc3, albedo_8uc3, CV_BGR2HSV);
			vector<Mat> _splt; split(albedo_8uc3,_splt);
			Mat eqd;
			equalizeHist(_splt[2], eqd); //eq-hist of Value channel
			_splt[2] = eqd * 0.55 + _splt[2] * 0.45;
			merge(_splt,albedo_8uc3);
			cvtColor(albedo_8uc3, albedo_8uc3, CV_HSV2BGR);

			//diff between albedo and original
			roi = _albedo_and_original(Rect(Point(albedo.cols*2,0),albedo.size()));
			Mat diff = albedo_8uc3 - face_img;
			diff.convertTo(roi, CV_32F, 1.0/255.0);

			albedo_8uc3.convertTo(albedo, CV_32FC3, 1.0/255.0);
		}
#endif
		roi = _albedo_and_original(Rect(Point(0,0),albedo.size()));
		albedo.copyTo(roi);
		
		//		putText(roi, "Computed", Point(20,20), CV_FONT_HERSHEY_DUPLEX, 0.8, Scalar(255));

		roi = _albedo_and_original(Rect(Point(albedo.cols,0),face_img.size()));
		face_img.convertTo(roi, CV_32F, 1.0/255.0);
		putText(roi, "Original", Point(20,20), CV_FONT_HERSHEY_DUPLEX, 0.8, Scalar(255));

//		roi = _albedo_and_original(Rect(Point(albedo.cols+face_img_hsv.cols,0),face_img_hsv.size()));
//		vector<Mat> albedo_and_original; split(face_img_hsv, albedo_and_original);
//		albedo.convertTo(albedo_and_original[2],CV_8U,255.0);
//		merge(albedo_and_original,face_img_hsv);
//		Mat _tmp; cvtColor(face_img_hsv, _tmp, CV_HSV2BGR); _tmp.convertTo(roi, CV_32F, 1.0/255.0);

//		merge(albedo_and_original,_albedo_and_original);
//		Mat _tmp; cvtColor(_albedo_and_original, _tmp, CV_HSV2BGR);
		imshow("albedo",_albedo_and_original); waitKey(1);
		if(_debug) waitKey(0);
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "computeAlbedo: " << t <<"s"<< endl;
	}
	
	void poissonBlendRecoverCompleteFace() {
		
	}
};
