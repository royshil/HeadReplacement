#include <opencv2/opencv.hpp>
using namespace cv;

#include "Recoloring.h"



#include <iostream>
#include <limits>

namespace VirtualSurgeon {

	Recoloring::~Recoloring(void)
	{
	}

	vector<int> Recoloring::MatchGaussians(CvEM& source_model, CvEM& target_model) {
		int num_g = source_model.get_nclusters();
		Mat sMu(source_model.get_means());
		Mat tMu(target_model.get_means());
		const CvMat** target_covs = target_model.get_covs();
		const CvMat** source_covs = source_model.get_covs();

		double best_dist = std::numeric_limits<double>::max();
		vector<int> best_res(num_g);
		vector<int> prmt(num_g); 

		for(int itr = 0; itr < 10; itr++) {
			for(int i=0;i<num_g;i++) prmt[i] = i;	//make a permutation
			randShuffle(prmt);

			//Greedy selection
			vector<int> res(num_g);
			vector<bool> taken(num_g);
			for(int sg = 0; sg < num_g; sg++) {
				double min_dist = std::numeric_limits<double>::max(); 
				int minv = -1;
				for(int tg = 0; tg < num_g; tg++) {
					if(taken[tg]) continue;

					//TODO: can save on re-calculation of pairs - calculate affinity matrix ahead
					//double d = norm(sMu(Range(prmt[sg],prmt[sg]+1),Range(0,3)),	tMu(Range(tg,tg+1),Range(0,3)));
					
					//symmetric kullback-leibler
					Mat diff = Mat(sMu(Range(prmt[sg],prmt[sg]+1),Range(0,3)) - tMu(Range(tg,tg+1),Range(0,3)));
					Mat d = diff * Mat(Mat(source_covs[prmt[sg]]).inv() + Mat(target_covs[tg]).inv()) * diff.t();
					Scalar tr = trace(Mat(
						Mat(Mat(source_covs[prmt[sg]])*Mat(target_covs[tg])) + 
						Mat(Mat(target_covs[tg])*Mat(source_covs[prmt[sg]]).inv()) + 
						Mat(Mat::eye(3,3,CV_64FC1)*2)
						));
					double kl_dist = ((double*)d.data)[0] + tr[0];
					if(kl_dist<min_dist) {
						min_dist = kl_dist;
						minv = tg;
					}
				}
				res[prmt[sg]] = minv;
				taken[minv] = true;
			}

			double dist = 0;
			for(int i=0;i<num_g;i++) {
				dist += norm(sMu(Range(prmt[i],prmt[i]+1),Range(0,3)),
							tMu(Range(res[prmt[i]],res[prmt[i]]+1),Range(0,3)));
			}
			if(dist < best_dist) {
				best_dist = dist;
				best_res = res;
			}
		}

		return best_res;
	}

	void Recoloring::TrainGMM(CvEM& source_model, Mat& source, Mat& source_mask) {
		int src_samples_size = countNonZero(source_mask);
		Mat source_samples(src_samples_size,3,CV_32FC1);

		Mat source_32f; 
		//if(source.type() != CV_32F)
		//	source.convertTo(source_32f,CV_32F,1.0/255.0);
		//else
			source_32f = source;

		int sample_count = 0;
		for(int y=0;y<source_32f.rows;y++) {
			Vec3f* row = source_32f.ptr<Vec3f>(y);
			uchar* mask_row = source_mask.ptr<uchar>(y);
			for(int x=0;x<source.cols;x++) {
				if(mask_row[x] > 0) {
					source_samples.at<Vec3f>(sample_count++,0) = row[x];
				}
			}
		}

		source_model.clear();
		CvEMParams ps(1/* = number of gaussians*/);
		source_model.train(source_samples,Mat(),ps,NULL);
	}

	void ChangeColors(CvEM& source_model,CvEM& target_model, Mat& target_32f, Mat& target_mask, vector<int>& match, int channels[]) {
		const CvMat** target_covs = target_model.get_covs();
		const CvMat** source_covs = source_model.get_covs();
		Mat sMu(source_model.get_means()); Mat sMu_64f; sMu.convertTo(sMu_64f,CV_64F);
		Mat tMu(target_model.get_means()); Mat tMu_64f; tMu.convertTo(tMu_64f,CV_64F);

		int num_g = target_model.get_nclusters();

		Mat pr; Mat samp(1,3,CV_32FC1);
		for(int y=0;y<target_32f.rows;y++) {
			Vec3f* row = target_32f.ptr<Vec3f>(y);
			uchar* mask_row = target_mask.ptr<uchar>(y);
			for(int x=0;x<target_32f.cols;x++) {
				if(mask_row[x] > 0) {
					memcpy(samp.data,&(row[x][0]),3*sizeof(float)); 
					float res = target_model.predict(samp,&pr);
					
					//cout << res << ":" << ((float*)pr.data)[0] << "," <<
					//	((float*)pr.data)[1] << "," <<
					//	((float*)pr.data)[2] << "," <<
					//	((float*)pr.data)[3] << "," <<
					//	((float*)pr.data)[4] << "," << endl;

					Mat samp_64f; samp.convertTo(samp_64f,CV_64F);

					//From Shapira09: Xnew = Sum_i { pr(i) * Sigma_source_i * (Sigma_target_i)^-1 * (x - mu_target) + mu_source }
					Mat Xnew(1,3,CV_64FC1,Scalar(0));
					for(int i=0;i<num_g;i++) {
						if(((float*)pr.data)[i] <= 0) continue;
						Mat A = /*Mat(*/
							//Mat(target_covs[i]) * 
							//Mat(source_covs[match[i]]).inv() *
							Mat(samp_64f - tMu_64f(Range(i,i+1),Range(0,3))).t();
						Mat B = sMu_64f(Range(match[i],match[i]+1),Range(0,3)).t();
						Xnew += Mat((A + B) * (double)(((float*)pr.data)[i])).t();
					}

					Mat _tmp; Xnew.convertTo(_tmp,CV_32F);

					//allow for mask with alpha
					float* _d = ((float*)_tmp.data);
					float alpha = mask_row[x] / 255.0f;

					for(int cl=0;cl<3;cl++) {
						if(channels[cl] == 1)	//only copy certain channels
							_d[cl] = _d[cl] * (alpha) + row[x][cl] * (1-alpha);
						else
							_d[cl] = row[x][cl];
					}
					
					memcpy(&(row[x][0]),_tmp.data,sizeof(float)*3);
				}
			}
		}
	}

	void Recoloring::Recolor(Mat& _source, Mat& source_mask, Mat& _target, Mat& target_mask, Mat alt_source_mask) {
		Mat source,__source; _source.convertTo(__source,CV_32F,1.0/255.0);
		Mat target,__target; _target.convertTo(__target,CV_32F,1.0/255.0);

		CvEM source_model,target_model;

		cvtColor(__target,target,CV_BGR2Lab);
		cvtColor(__source,source,CV_BGR2Lab);

		if(!alt_source_mask.empty())
			TrainGMM(source_model,source,alt_source_mask);
		else 
			TrainGMM(source_model,source,source_mask);
		TrainGMM(target_model,target,target_mask);

		vector<int> match = MatchGaussians(source_model,target_model);
		

		Mat target_32f; 
		//if(target.type() != CV_32F)
		//	target.convertTo(target_32f,CV_32F,1.0/255.0);
		//else
		target.copyTo(target_32f);

		if(m_p.two_way_recolor) {
			//Copy \alpha \beta from source to target
			int cls[3] = {0,1,1};
			ChangeColors(source_model,target_model,target_32f,target_mask,match,cls);
		
			//Copy Luma from target to source
			cls[0] = 1; cls[1] = 0; cls[2] = 0;
			ChangeColors(target_model,source_model,source,source_mask,match,cls);
		} else {
			//one way recoloring - everything goes from source to target
			int cls[3] = {1,1,1};
			ChangeColors(source_model,target_model,target_32f,target_mask,match,cls);
		}
	
		cvtColor(target,target,CV_Lab2BGR);
		cvtColor(source,source,CV_Lab2BGR);
		cvtColor(target_32f,target_32f,CV_Lab2BGR);

		if(!this->m_p.no_gui) {
			imshow("orig target",target);
			imshow("source changed",source);
			imshow("dest target",target_32f);

			waitKey(this->m_p.wait_time);
		}

		target_32f.convertTo(_target,CV_8UC3,255.0);
		source.convertTo(_source,CV_8UC3,255.0);
	}

}//ns

/*
int main(int argc, char** argv) {
	VirtualSurgeon::VirtualSurgeonParams p;
	p.InitializeDefault();
	p.no_gui = false;
	p.wait_time = 0;
	VirtualSurgeon::Recoloring r(p);
	
	Mat dst = imread("../images/454867616_1c0006f61a_o_d.model.png");
	Mat dst_mask = imread("../images/454867616_1c0006f61a_o_d.skin_mask.png",0);
	Mat src = imread("../images/2402378006_c5b9bb75c9_o_d.model.png");
	Mat src_mask = imread("../images/2402378006_c5b9bb75c9_o_d.skin_mask.png",0);

	Mat _tmp;
	double s = 0.5;
	resize(dst,_tmp,Size(),s,s); _tmp.copyTo(dst);
	resize(dst_mask,_tmp,Size(),s,s,INTER_NEAREST); _tmp.copyTo(dst_mask);
	resize(src,_tmp,Size(),s,s); _tmp.copyTo(src);
	resize(src_mask,_tmp,Size(),s,s,INTER_NEAREST); _tmp.copyTo(src_mask);

	r.Recolor(src,src_mask,dst,dst_mask);

}
*/