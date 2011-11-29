#include "matting.h"

#ifdef ROBUST_MATTING
#include <algorithm>

#include "solver.h"

#include "mtl/matrix.h"
//#include "mtl/mtl.h"
//#include "mtl/utils.h"
//#include "mtl/linalg_vec.h"
//#include "mtl/lu.h"
//#include "mtl/dense1D.h"


using namespace mtl;

typedef matrix< float, 
                symmetric<upper>, 
	            array< compressed<> >, 
                row_major >::type SymMatrix;

typedef matrix<float, 
			mtl::rectangle<>, 
             array< compressed<> >, 
             row_major>::type Matrix;
#endif

#ifdef POISSON_MATTING
#include <map>
using std::map;

#include "mkl_dss.h"
#endif





/* ==================================================================================
	Bayesian Matting
   ================================================================================== */


BayesianMatting::BayesianMatting( IplImage* cImg, IplImage* tmap )
{
	colorImg = cvCloneImage( cImg );
	trimap   = cvCloneImage( tmap );	
	
	Initialize();
	SetParameter();
}

BayesianMatting::~BayesianMatting()
{
	if(colorImg)
		cvReleaseImage( &colorImg );
	if(fgImg)
		cvReleaseImage( &fgImg );
	if(bgImg)
		cvReleaseImage( &bgImg );
	if(trimap)
		cvReleaseImage( &trimap );
	if(alphamap)
		cvReleaseImage( &alphamap );	
	if(fgmask)
		cvReleaseImage( &fgmask );
	if(bgmask)
		cvReleaseImage( &bgmask );
	if(unmask)
		cvReleaseImage( &unmask );
	if(unsolvedmask)
		cvReleaseImage( &unmask );
}

void BayesianMatting::Initialize()
{
	fgImg    = cvCreateImage( cvGetSize( trimap ), 8, 3 );
	bgImg    = cvCreateImage( cvGetSize( trimap ), 8, 3 );
	fgmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	bgmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	unmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	unsolvedmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	alphamap = cvCreateImage( cvGetSize( colorImg ), 32, 1 );	

	cvZero( fgImg );
	cvZero( bgImg );
	cvZero( fgmask );
	cvZero( bgmask );
	cvZero( unmask );
	cvZero( alphamap );
	//cvZero( unsolvedmask );	

	int i, j, v;
	for(i=0;i<trimap->height;++i)
		for(j=0;j<trimap->width;++j)
		{
			v = CV_IMAGE_ELEM( trimap, uchar, i, j );
			if( v == BACKGROUND_VALUE )
				CV_IMAGE_ELEM( bgmask, uchar, i, j ) = 255;							
			else if( v == FOREGROUND_VALUE )			
				CV_IMAGE_ELEM( fgmask, uchar, i, j ) = 255;			
			else
				CV_IMAGE_ELEM( unmask, uchar, i, j ) = 255;			
		}

	cvSet( alphamap, cvScalarAll( 0 ), bgmask );
	cvSet( alphamap, cvScalarAll( 1 ), fgmask );
	cvCopyImage( unmask, unsolvedmask );
	cvCopy( colorImg, fgImg, fgmask );
	cvCopy( colorImg, bgImg, bgmask );
}

void BayesianMatting::SetParameter( int N, float sigma_, float sigma_c )
{
	nearest = N;
	sigmac = sigma_c;
	sigma  = sigma_;	
}

double BayesianMatting::Solve(bool do_gui = false)
{
	int p, r, c, i, j, iter, fgClus, bgClus;
	int outter;
	float L, maxL;

	IplImage* shownImg       = cvCreateImage( cvGetSize(colorImg), 8, 3 );
	IplImage* solveAgainMask = cvCreateImage( cvGetSize( unmask ), 8, 1 );

	vector<float>   fg_weight( BAYESIAN_MAX_CLUS, 0 );
	vector<float>   bg_weight( BAYESIAN_MAX_CLUS, 0 );
	vector<CvMat *> fg_mean( BAYESIAN_MAX_CLUS );
	vector<CvMat *> bg_mean( BAYESIAN_MAX_CLUS );
	vector<CvMat *> inv_fg_cov( BAYESIAN_MAX_CLUS );
	vector<CvMat *> inv_bg_cov( BAYESIAN_MAX_CLUS );
	for(i=0;i<BAYESIAN_MAX_CLUS;i++)
	{
		fg_mean[i]    = cvCreateMat(3,1,CV_32FC1);
		bg_mean[i]    = cvCreateMat(3,1,CV_32FC1);
		inv_fg_cov[i] = cvCreateMat(3,3,CV_32FC1);
		inv_bg_cov[i] = cvCreateMat(3,3,CV_32FC1);
	}

	for(int Iteration = 0 ; Iteration<1; ++Iteration )
	{
		printf("\niteration %d:\n", Iteration);

		if(Iteration)
			cvCopy( unmask, solveAgainMask );

		outter = 0;
		for(;;)
		{				
			printf("solving contour %d\r", outter++);

			vector<CvPoint> toSolveList;

			if(!Iteration)
				GetContour( unsolvedmask, toSolveList );
			else
				GetContour( solveAgainMask, toSolveList );
			
			//no unknown left
			if( !toSolveList.size() )
				break;

			if(do_gui) {
				cvCopyImage( colorImg, shownImg );
				for(int k=0;k<toSolveList.size();++k)
					cvCircle( shownImg, toSolveList[k], 1, cvScalarAll( 128 ) );
				cvNamedWindow( "points to solve" );
				cvShowImage( "points to solve", shownImg );
				cvMoveWindow( "points to solve", 0, 0 );
				cvWaitKey( 1 );
			}


			//solve the points in the list one by one
			for( p = 0 ; p < toSolveList.size() ;++p )
			{	
				r = toSolveList[p].y, c = toSolveList[p].x;

				//get the gmm model using the neighbors of foreground and neighbors of background			
				GetGMMModel( r, c, fg_weight, fg_mean, inv_fg_cov, bg_weight, bg_mean, inv_bg_cov);
							
				maxL = (float)-INT_MAX;

				for(i=0;i<BAYESIAN_MAX_CLUS;i++)
					for(j=0;j<BAYESIAN_MAX_CLUS;j++)
					{
						//initilize the alpha by the average of near points
						if(!Iteration)
							InitializeAlpha( r, c, unsolvedmask );
						else
							InitializeAlpha( r, c, solveAgainMask );

						for(iter=0;iter<3;++iter)
						{
							SolveBF( r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j] );
							SolveAlpha( r, c );
						}

						// largest likelihood, restore the index in fgClus, bgClus
						L = computeLikelihood( r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j]);
						//L = computeLikelihood( r, c, fg_weight[i], fg_mean[i], inv_fg_cov[i], bg_weight[j], bg_mean[j], inv_bg_cov[j]);
						if(L>maxL)
						{
							maxL = L;
							fgClus = i;
							bgClus = j;
						}
					}
							
				
				if(!Iteration)
					InitializeAlpha( r, c, unsolvedmask );
				else
					InitializeAlpha( r, c, solveAgainMask );

				for(iter=0;iter<5;++iter)
				{
					SolveBF( r, c, fg_mean[fgClus], inv_fg_cov[fgClus], bg_mean[bgClus], inv_bg_cov[bgClus] );
					SolveAlpha( r, c );
				}
				//printf("%f\n", CV_IMAGE_ELEM(alphamap,float,r,c));

				//solved!
				if(!Iteration)
					CV_IMAGE_ELEM( unsolvedmask, uchar, r, c ) = 0;
				else
					CV_IMAGE_ELEM( solveAgainMask, uchar, r, c ) = 0;
			}
			//cvNamedWindow("fg");
			//cvShowImage("fg", fgImg );
			//cvMoveWindow("fg",0,100+colorImg->height);
			//cvNamedWindow("bg");
			//cvShowImage("bg", bgImg );
			//cvMoveWindow("bg",100+colorImg->width,100+colorImg->height);
			if(do_gui) {
				cvNamedWindow("alphamap");
				cvShowImage("alphamap", alphamap );
				cvMoveWindow("alphamap",100+colorImg->width, 0);
				cvWaitKey( 1 );
			}
		}
	}

	printf("\nDone!!\n");

	/////////////////////////

	cvReleaseImage( &shownImg );
	cvReleaseImage( &solveAgainMask );

	for(i=0;i<fg_mean.size();i++)
	{
		cvReleaseMat( &fg_mean[i] );
		cvReleaseMat( &bg_mean[i] );
		cvReleaseMat( &inv_fg_cov[i] );
		cvReleaseMat( &inv_bg_cov[i] );
	}
	return 1;
}

void BayesianMatting::InitializeAlpha( int r, int c, const IplImage* unSolvedMask )
{
	int i, j;
	int min_x, min_y, max_x, max_y;
#define WIN_SIZE 1


	min_x = max(0, c - WIN_SIZE);
	min_y = max(0, r - WIN_SIZE);
	max_x = min(colorImg->width - 1, c + WIN_SIZE );
	max_y = min(colorImg->height - 1, r + WIN_SIZE );

	int count = 0;
	float sum = 0;
	for( i = min_y; i<=max_y; ++i )
		for( j = min_x; j<=max_x; ++j )
		{
			if( !CV_IMAGE_ELEM(unSolvedMask, uchar, i, j) )
			{
				sum += CV_IMAGE_ELEM(alphamap, float, i, j);
				++count;
			}
		}
	
	CV_IMAGE_ELEM( alphamap, float, r, c ) = (count? sum / count : 0);	
}

void BayesianMatting::GetContour( IplImage* img, vector<CvPoint> &contour )
{
	contour.clear();

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;

	cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

	for( ; contours != 0; contours = contours->h_next )
	{
		CvSeqReader reader;		
		cvStartReadSeq( contours, &reader, 0 );
		
		int i, count = contours->total;
		
		CvPoint pt;
		for( i = 0; i < count; i++ )
        {       
            CV_READ_SEQ_ELEM( pt, reader );
			contour.push_back( cvPoint( pt.x, pt.y ) );
        }
	}

	cvReleaseMemStorage( &storage );	
}

void BayesianMatting::CalculateNonNormalizeCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov )
{
	int cur_r, cur_c;
	float cur_w, total_w=0;
	cvZero( mean );
	cvZero( cov );
	for(size_t j=0;j<clus_set.size();j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for(int h=0;h<3;h++)
		{
			CV_MAT_ELEM( *mean, float, h, 0 ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h ));
			for(int k=0;k<3;k++)
				CV_MAT_ELEM( *cov, float, h, k ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h )*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+k ));
		}

		total_w += clus_set[j].second;
	}
			
	float inv_total_w = 1.f/total_w;
	for(int h=0;h<3;h++)
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) -= (inv_total_w*CV_MAT_ELEM( *mean, float, h, 0 )*CV_MAT_ELEM( *mean, float, k, 0 ));

}

void BayesianMatting::CalculateMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov )
{
	int cur_r, cur_c;
	float cur_w, total_w=0;
	cvZero( mean );
	cvZero( cov );
	for(size_t j=0;j<clus_set.size();j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for(int h=0;h<3;h++)
		{
			CV_MAT_ELEM( *mean, float, h, 0 ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h ));
			for(int k=0;k<3;k++)
				CV_MAT_ELEM( *cov, float, h, k ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h )*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+k ));
		}

		total_w += clus_set[j].second;
	}
			
	float inv_total_w = 1.f/total_w;
	for(int h=0;h<3;h++)
	{
		CV_MAT_ELEM( *mean, float, h, 0 ) *= inv_total_w;
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) *= inv_total_w;
	}

	for(int h=0;h<3;h++)
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) -= (CV_MAT_ELEM( *mean, float, h, 0 )*CV_MAT_ELEM( *mean, float, k, 0 ));
}

void BayesianMatting::CalculateWeightMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, float &weight, CvMat* mean, CvMat* cov )
{
	int cur_r, cur_c;
	float cur_w, total_w=0;
	cvZero( mean );
	cvZero( cov );
	for(size_t j=0;j<clus_set.size();j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for(int h=0;h<3;h++)
		{
			CV_MAT_ELEM( *mean, float, h, 0 ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h ));
			for(int k=0;k<3;k++)
				CV_MAT_ELEM( *cov, float, h, k ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h )*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+k ));
		}

		total_w += clus_set[j].second;
	}
			
	float inv_total_w = 1.f/total_w;
	for(int h=0;h<3;h++)
	{
		CV_MAT_ELEM( *mean, float, h, 0 ) *= inv_total_w;
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) *= inv_total_w;
	}

	for(int h=0;h<3;h++)
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) -= (CV_MAT_ELEM( *mean, float, h, 0 )*CV_MAT_ELEM( *mean, float, k, 0 ));

	weight = total_w;
}


void BayesianMatting::GetGMMModel( int r, int c, vector<float> &fg_weight, const vector<CvMat*> fg_mean, const vector<CvMat*> inv_fg_cov, vector<float> &bg_weight, const vector<CvMat*> bg_mean, const vector<CvMat*> inv_bg_cov )
{
	vector<pair<CvPoint,float> > fg_set, bg_set;
	CollectSampleSet( r, c, fg_set, bg_set );

	//IplImage* tmp1 = cvCloneImage( colorImg );	
	//IplImage* tmp2 = cvCloneImage( colorImg );	
	//			
	//for(size_t i=0;i<fg_set.size();++i)
	//{
	//	cvCircle( tmp1, fg_set[i].first, 1, cvScalar( 0, 0, fg_set[i].second * 255 ) );	
	//	cvCircle( tmp2, bg_set[i].first, 1, cvScalar( bg_set[i].second * 255, 0, 0 ) );	
	//}

	//cvNamedWindow( "fg_sample" );
	//cvShowImage( "fg_sample", tmp1 );
	//cvNamedWindow( "bg_sample" );
	//cvShowImage( "bg_sample", tmp2 );
	//cvWaitKey( 0 );
	//cvReleaseImage( &tmp1 );	
	//cvReleaseImage( &tmp2 );	

	CvMat *mean = cvCreateMat(3,1,CV_32FC1);
	CvMat *cov = cvCreateMat(3,3,CV_32FC1);
	CvMat *inv_cov = cvCreateMat(3,3,CV_32FC1);
	CvMat *eigval = cvCreateMat(3,1,CV_32FC1);
	CvMat *eigvec = cvCreateMat(3,3,CV_32FC1);
	CvMat *cur_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *max_eigvec = cvCreateMat(3,1,CV_32FC1);
	CvMat *target_color = cvCreateMat(3,1,CV_32FC1);
//fg

	//// initializtion
	vector<pair<CvPoint,float> > clus_set[BAYESIAN_MAX_CLUS];
	int nClus = 1;
	clus_set[0] = fg_set;

	while(nClus<BAYESIAN_MAX_CLUS)
	{
		// find the largest eigenvalue
		double max_eigval = 0;
		int max_idx = 0;
		for(int i=0;i<nClus;i++)
		{
			//CalculateMeanCov(clus_set[i],mean,cov);
			CalculateNonNormalizeCov(fgImg,clus_set[i],mean,cov);

			// compute eigval, and eigvec
			cvSVD(cov, eigval, eigvec);
			if(cvmGet(eigval,0,0)>max_eigval)
			{
				cvGetCol(eigvec,max_eigvec,0);
				max_eigval = cvmGet(eigval,0,0);
				max_idx = i;
			}
		}

		// split
		vector<pair<CvPoint,float> > new_clus_set[2];
		CalculateMeanCov(fgImg,clus_set[max_idx],mean,cov);
		double boundary = cvDotProduct(mean,max_eigvec);
		for(size_t i=0;i<clus_set[max_idx].size();i++)
		{
			for(int j=0;j<3;j++)
				cvmSet(cur_color,j,0,CV_IMAGE_ELEM( fgImg, uchar, clus_set[max_idx][i].first.y, 3*clus_set[max_idx][i].first.x+j ));
			
			if(cvDotProduct(cur_color,max_eigvec)>boundary)
				new_clus_set[0].push_back(clus_set[max_idx][i]);
			else
				new_clus_set[1].push_back(clus_set[max_idx][i]);
		}

		clus_set[max_idx] = new_clus_set[0];
		clus_set[nClus] = new_clus_set[1];

		nClus += 1;
	}

	// return all the mean and cov of fg
	float weight_sum, inv_weight_sum;
	weight_sum = 0;
	for(int i=0;i<nClus;i++)
	{
		CalculateWeightMeanCov(fgImg,clus_set[i],fg_weight[i],fg_mean[i],cov);
		cvInvert(cov,inv_fg_cov[i]);
		weight_sum += fg_weight[i];
	}
	//normalize weight
	inv_weight_sum = 1.f / weight_sum;
	for(int i=0;i<nClus;i++)
		fg_weight[i] *= inv_weight_sum;

// bg
	// initializtion
	nClus = 1;
	for(int i=0;i<BAYESIAN_MAX_CLUS;++i)
		clus_set[i].clear();
	clus_set[0] = bg_set;

	while(nClus<BAYESIAN_MAX_CLUS)
	{
		// find the largest eigenvalue
		double max_eigval = 0;
		int max_idx = 0;
		for(int i=0;i<nClus;i++)
		{
			//CalculateMeanCov(clus_set[i],mean,cov);
			CalculateNonNormalizeCov(bgImg,clus_set[i],mean,cov);

			// compute eigval, and eigvec
			cvSVD(cov, eigval, eigvec);
			if(cvmGet(eigval,0,0)>max_eigval)
			{
				cvGetCol(eigvec,max_eigvec,0);
				max_eigval = cvmGet(eigval,0,0);
				max_idx = i;
			}
		}

		// split
		vector<pair<CvPoint,float> > new_clus_set[2];
		CalculateMeanCov(bgImg,clus_set[max_idx],mean,cov);
		double boundary = cvDotProduct(mean,max_eigvec);
		for(size_t i=0;i<clus_set[max_idx].size();i++)
		{
			for(int j=0;j<3;j++)
				cvmSet(cur_color,j,0,CV_IMAGE_ELEM( bgImg, uchar, clus_set[max_idx][i].first.y, 3*clus_set[max_idx][i].first.x+j ));
			
			if(cvDotProduct(cur_color,max_eigvec)>boundary)
				new_clus_set[0].push_back(clus_set[max_idx][i]);
			else
				new_clus_set[1].push_back(clus_set[max_idx][i]);
		}

		clus_set[max_idx] = new_clus_set[0];
		clus_set[nClus] = new_clus_set[1];

		nClus += 1;
	}

	// return all the mean and cov of bg
	weight_sum = 0;
	for(int i=0;i<nClus;i++)
	{
		CalculateWeightMeanCov(bgImg,clus_set[i],bg_weight[i],bg_mean[i],cov);
		cvInvert(cov,inv_bg_cov[i]);
		weight_sum += bg_weight[i];
	}
	//normalize weight
	inv_weight_sum = 1.f / weight_sum;
	for(int i=0;i<nClus;i++)
		bg_weight[i] *= inv_weight_sum;
	
	cvReleaseMat( &mean ), cvReleaseMat( &cov ), cvReleaseMat( &eigval ), cvReleaseMat( &eigvec ), cvReleaseMat( &cur_color ), cvReleaseMat( &inv_cov ), cvReleaseMat( &max_eigvec ), cvReleaseMat( &target_color );
}

void BayesianMatting::CollectSampleSet( int r, int c, vector<pair<CvPoint, float> > &fg_set, vector<pair<CvPoint, float> > &bg_set )
{	
	fg_set.clear(), bg_set.clear();	
#define UNSURE_DISTANCE 1	
	
	pair<CvPoint, float> sample;
	float dist_weight;
	float inv_2sigma_square = 1.f / (2*sigma*sigma);

	int dist = 1;
	while(fg_set.size() < nearest)
	{		
		if( r - dist >= 0 )
		{
			for(int z = max(0, c - dist); z<= min(colorImg->width-1, c+dist); ++z )
			{ 
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r-dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r-dist, z) * CV_IMAGE_ELEM(alphamap, float, r-dist, z) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( r + dist < colorImg->height)
		{			
			for(int z = max(0, c - dist); z<= min(colorImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r+dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r+dist, z) * CV_IMAGE_ELEM(alphamap, float, r+dist, z) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( c - dist >= 0)
		{			
			for(int z = max(0, r - dist+ 1); z<= min(colorImg->height-1, r+dist - 1 ); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c - dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c-dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c - dist) * CV_IMAGE_ELEM(alphamap, float, z, c - dist) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( c + dist < colorImg->width )
		{
			for(int z = max(0, r - dist + 1); z<= min(colorImg->height-1, r + dist - 1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c + dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c+dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c + dist) * CV_IMAGE_ELEM(alphamap, float, z, c + dist) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		++dist;
	}

BG:
	int bg_unsure = 0;
	dist = 1;

	while(bg_set.size() < nearest)
	{
		dist_weight = expf( -(dist*dist)/(2*sigma*sigma) );
		if( r - dist >= 0 )
		{
			for(int z = max(0, c - dist); z<= min(colorImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r-dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r-dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r-dist, z)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( r + dist < colorImg->height)
		{
			for(int z = max(0, c - dist); z<= min(colorImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r+dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r+dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r+dist, z)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( c - dist >= 0)
		{
			for(int z = max(0, r - dist + 1); z<= min(colorImg->height-1, r+dist-1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c-dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( c + dist < colorImg->width )
		{
			for(int z = max(0, r - dist + 1); z<= min(colorImg->height-1, r+dist-1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c+dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		++dist;
	}

DONE:
	assert( fg_set.size() == nearest );
	assert( bg_set.size() == nearest );
}

void BayesianMatting::SolveBF( int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov )
{
	CvMat *A	= cvCreateMat( 6, 6, CV_32FC1 );
	CvMat *x	= cvCreateMat( 6, 1, CV_32FC1 );
	CvMat *b	= cvCreateMat( 6, 1, CV_32FC1 );
	CvMat *I	= cvCreateMat( 3, 3, CV_32FC1 );
	CvMat *work_3x3 = cvCreateMat( 3, 3, CV_32FC1 );
	CvMat *work_3x1 = cvCreateMat( 3, 1, CV_32FC1 );
	
	float alpha = CV_IMAGE_ELEM( alphamap, float, r, c );
	CvScalar fg_color = cvScalar( CV_IMAGE_ELEM( fgImg, uchar, r, 3*c ), CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 ), CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 ));
	CvScalar bg_color = cvScalar( CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ), CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ), CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 ));
	CvScalar  c_color = cvScalar( CV_IMAGE_ELEM( colorImg, uchar, r, 3*c ), CV_IMAGE_ELEM( colorImg, uchar, r, 3*c+1 ), CV_IMAGE_ELEM( colorImg, uchar, r, 3*c+2 ));

	float inv_sigmac_square = 1.f/(sigmac*sigmac);

	cvZero( I );
	CV_MAT_ELEM( *I, float, 0, 0 ) = CV_MAT_ELEM( *I, float, 1, 1 ) = CV_MAT_ELEM( *I, float, 2, 2 ) = 1.f;

	////a
	cvCvtScale( I, work_3x3, alpha*alpha*inv_sigmac_square );
	cvAdd( inv_fg_cov, work_3x3, work_3x3 );
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, i, j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	//
	cvCvtScale( I, work_3x3, alpha*(1-alpha)*inv_sigmac_square);
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, i, 3+j ) = CV_MAT_ELEM( *A, float, 3+i, j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	//
	cvCvtScale( I, work_3x3, (1-alpha)*(1-alpha)*inv_sigmac_square );
	cvAdd( inv_bg_cov, work_3x3, work_3x3 );
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, 3+i, 3+j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	////x
	cvZero( x );

	////b
	cvMatMul( inv_fg_cov, fg_mean, work_3x1 );
	for(int i=0;i<3;++i)
		CV_MAT_ELEM( *b, float, i, 0 ) = CV_MAT_ELEM( *work_3x1, float, i, 0 ) + (float)c_color.val[i]*alpha*inv_sigmac_square;
	//
	cvMatMul( inv_bg_cov, bg_mean, work_3x1 );
	for(int i=0;i<3;++i)
		CV_MAT_ELEM( *b, float, 3+i, 0 ) = CV_MAT_ELEM( *work_3x1, float, i, 0 ) + (float)c_color.val[i]*(1-alpha)*inv_sigmac_square;


	//
	cvSolve( A, b, x );
	
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )   = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 0, 0 )));
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 1, 0 )));
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 2, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c )   = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 3, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 4, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 5, 0 )));

	cvReleaseMat( &A ), cvReleaseMat( &x ), cvReleaseMat( &b ), cvReleaseMat( &I ), cvReleaseMat( &work_3x3 ), cvReleaseMat( &work_3x1 );
}

inline void BayesianMatting::SolveAlpha(int r, int c)
{
	CV_IMAGE_ELEM( alphamap, float, r, c ) =
	 (	 ((float)CV_IMAGE_ELEM( colorImg, uchar, r, 3*c )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))   * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )     - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))
	   + ((float)CV_IMAGE_ELEM( colorImg, uchar, r, 3*c+1 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ))
	   + ((float)CV_IMAGE_ELEM( colorImg, uchar, r, 3*c+2 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 ))
	 ) / 
	 (
		 ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))   * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )     - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))
	   + ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ))
	   + ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 ))
	 );

	CV_IMAGE_ELEM( alphamap, float, r, c ) = MAX( 0, MIN( 1, CV_IMAGE_ELEM( alphamap, float, r, c )));
}

float BayesianMatting::computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov)
{
	float fgL, bgL, cL;
	int i;
	float alpha = CV_IMAGE_ELEM( alphamap, float, r, c );

	CvMat *work3x1 = cvCreateMat(3,1,CV_32FC1);
	CvMat *work1x3 = cvCreateMat(1,3,CV_32FC1);
	CvMat *work1x1 = cvCreateMat(1,1,CV_32FC1);
	CvMat *fg_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *bg_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *c_color = cvCreateMat(3,1,CV_32FC1);
	for(i=0;i<3;i++)
	{	
		CV_MAT_ELEM( *fg_color, float, i,0) = CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+i );
		CV_MAT_ELEM( *bg_color, float, i,0) = CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+i );
		CV_MAT_ELEM( *c_color, float, i,0) = CV_IMAGE_ELEM( colorImg, uchar, r, 3*c+i );
	}

	// fgL
	cvSub(fg_color,fg_mean,work3x1);
	cvTranspose(work3x1,work1x3);
	cvMatMul(work1x3,inv_fg_cov,work1x3);
	cvMatMul(work1x3,work3x1,work1x1);
	fgL = -1.0f*CV_MAT_ELEM( *work1x1, float,0,0)/2;

	// bgL
	cvSub(bg_color,bg_mean,work3x1);
	cvTranspose(work3x1,work1x3);
	cvMatMul(work1x3,inv_bg_cov,work1x3);
	cvMatMul(work1x3,work3x1,work1x1);
	bgL = -1.f*CV_MAT_ELEM( *work1x1, float,0,0)/2;

	// cL
	cvAddWeighted(c_color, 1.0f, fg_color, -1.0f*alpha, 0.0f, work3x1 );
	cvAddWeighted(work3x1, 1.0f, bg_color, -1.0f*(1.0f-alpha), 0.0f, work3x1 );
	cL = -cvDotProduct( work3x1, work3x1 ) / (2 * sigmac * sigmac);

	cvReleaseMat( &work3x1 );
	cvReleaseMat( &work1x3 );
	cvReleaseMat( &work1x1 );
	cvReleaseMat( &fg_color );
	cvReleaseMat( &bg_color );
	cvReleaseMat( &c_color );

	return cL+fgL+bgL;
}

float BayesianMatting::computeLikelihood( int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov )
{
	return computeLikelihood( r, c, fg_mean, inv_fg_cov, bg_mean, inv_bg_cov ) + logf(fg_weight)+logf(bg_weight);
}


#ifdef ROBUST_MATTING

/* ==================================================================================
	Robust Matting
   ================================================================================== */

RobustMatting::RobustMatting( IplImage* cImg, IplImage* tmap )
{
	colorImg = cvCloneImage( cImg );
	trimap   = cvCloneImage( tmap );	
	
	Initialize();
	SetParameter();
}

RobustMatting::~RobustMatting()
{
	if(colorImg)
		cvReleaseImage( &colorImg );
	if(trimap)
		cvReleaseImage( &trimap );
	if(alphamap)
		cvReleaseImage( &alphamap );
	if(fgmask)
		cvReleaseImage( &fgmask );
	if(bgmask)
		cvReleaseImage( &bgmask );
	if(unmask)
		cvReleaseImage( &unmask );
	if(confidencemap)
		cvReleaseImage( &confidencemap );
}

void RobustMatting::Initialize()
{
	fgmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	bgmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	unmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	alphamap = cvCreateImage( cvGetSize( colorImg ), 32, 1 );
	confidencemap = cvCreateImage( cvGetSize( colorImg ), 32, 1 );

	cvZero( fgmask );
	cvZero( bgmask );
	cvZero( unmask );
	cvZero( alphamap );

	int i, j, v;
	for(i=0;i<trimap->height;++i)
		for(j=0;j<trimap->width;++j)
		{
			v = CV_IMAGE_ELEM( trimap, uchar, i, j );
			if( v == BACKGROUND_VALUE )			
				CV_IMAGE_ELEM( bgmask, uchar, i, j ) = 255;							
			else if( v == FOREGROUND_VALUE )			
				CV_IMAGE_ELEM( fgmask, uchar, i, j ) = 255;			
			else
				CV_IMAGE_ELEM( unmask, uchar, i, j ) = 255;
		}

	cvSet( alphamap, cvScalar(1), fgmask );
}

void RobustMatting::SetParameter( int Sample_Number , int Sample_Dist, float Sigma, float Gamma, float Etta )
{	
	sample_number = Sample_Number;
	sample_dist = Sample_Dist;
	sigma = Sigma;
	gamma = Gamma;
	etta = Etta;
}

void RobustMatting::GetContour( IplImage* img, vector<CvPoint> &contour )
{
	contour.clear();

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;

	cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

	for( ; contours != 0; contours = contours->h_next )
	{
		CvSeqReader reader;		
		cvStartReadSeq( contours, &reader, 0 );
		
		int i, count = contours->total;
		
		CvPoint pt;
		for( i = 0; i < count; i++ )
        {       
            CV_READ_SEQ_ELEM( pt, reader );
			contour.push_back( cvPoint( pt.x, pt.y ) );
        }
	}

	cvReleaseMemStorage( &storage );	
}

struct PointCandidate
{
	int index;
	float dist;
};

bool operator<(const PointCandidate& a, const PointCandidate& b) {
	return a.dist<b.dist;
}

void RobustMatting::CollectSampleSet( int r, int c, vector<CvPoint> &samp_set, const vector<CvPoint> &cand_set )
{	
	int i, d;
	int min_i = -1;
	int min_d = INT_MAX;
	
	samp_set.clear();

	vector<PointCandidate> pcs;
	//find the nearest point
	for(i=0;i<cand_set.size();++i)
	{
		PointCandidate pc;
		pc.index = i;
		pc.dist = (cand_set[i].y - r) * (cand_set[i].y - r) + (cand_set[i].x - c) * (cand_set[i].x - c);
		
		pcs.push_back( pc );
	}
	//sample along the contour
	std::sort( pcs.begin(), pcs.end() );

	for(i=0;i<sample_number;++i)
		samp_set.push_back( cand_set[pcs[i].index] );
}

void RobustMatting::BuildProblemMatrix( double *&A_value, int *&A_rowIndex, int *&A_columns, double *&b, double *&solution, int &nonzero, int &length, vector<CvPoint> &unknown_list )
{
	int i, j;
	int known, unknown;	
	vector<CvPoint> fg_list, bg_list;
	for(i=0;i<unmask->height;++i)
		for(j=0;j<unmask->width;++j)		
		{
			if(CV_IMAGE_ELEM(unmask, uchar, i, j))
				unknown_list.push_back( cvPoint( j, i ) );			
		}
	
	//Version 1, much faster
	GetContour( fgmask, fg_list );
	GetContour( bgmask, bg_list );

	//Version 2, much slower
	//for(i=0;i<unmask->height;++i)
	//	for(j=0;j<unmask->width;++j)		
	//	{			
	//		Version 1, much slower
	//		if(CV_IMAGE_ELEM(fgmask, uchar, i, j))
	//			fg_list.push_back( cvPoint( j, i ) );
	//		else if(CV_IMAGE_ELEM(bgmask, uchar, i, j))
	//			bg_list.push_back( cvPoint( j, i ) );
	//	}


	length = unknown = unknown_list.size();	
	known = fg_list.size() + bg_list.size() + 2;

	//intialize matrix
    SymMatrix Lu(unknown, unknown);
	Matrix Rt(unknown, known);

	//generate the Lu and Rt matrix in Laplacian matrix
	CvMat *mat_3x3 = cvCreateMat( 3, 3, CV_32FC1 );
	nonzero = 0;  //calculate the number of nonzero terms in the upper triangle. (include diagonal even when it is zero)
	float w_ij, row_sum;
	for(i=0;i<unknown;++i)
	{
		row_sum = 0;

		//Lu
		for(j=i+1;j<unknown;++j)
		{
			w_ij = SmoothWeight( unknown_list[i].y, unknown_list[i].x, unknown_list[j].y,  unknown_list[j].x, mat_3x3 );
			//w_ij = TestSmoothWeight( unknown_list[i].y, unknown_list[i].x, unknown_list[j].y,  unknown_list[j].x, mat_3x3 );
			if(w_ij)
			{				
				Lu( i, j ) = -w_ij;
				//Lu( j, i ) = -w_ij; //not necessary because it's symmetric matrix
				row_sum += (-w_ij);
				nonzero++;
			}
		}

		//Rt
		for(j=0;j<fg_list.size();++j)
		{			
			w_ij = SmoothWeight( unknown_list[i].y, unknown_list[i].x, fg_list[j].y,  fg_list[j].x, mat_3x3 );
			//w_ij = TestSmoothWeight( unknown_list[i].y, unknown_list[i].x, fg_list[j].y,  fg_list[j].x, mat_3x3 );
			if(w_ij)
			{
				Rt( i, j ) = -w_ij;
				row_sum += (-w_ij);
			}
		}
		for(j=0;j<bg_list.size();++j)
		{			
			w_ij = SmoothWeight( unknown_list[i].y, unknown_list[i].x, bg_list[j].y,  bg_list[j].x, mat_3x3 );
			//w_ij = TestSmoothWeight( unknown_list[i].y, unknown_list[i].x, bg_list[j].y,  bg_list[j].x, mat_3x3 );
			if(w_ij)
			{
				Rt( i, j + fg_list.size() ) = -w_ij;
				row_sum += (-w_ij);
			}
		}	
		Rt( i, known - 2 ) = - DataWeightFg( CV_IMAGE_ELEM( alphamap, float, unknown_list[i].y, unknown_list[i].x ), CV_IMAGE_ELEM( confidencemap, float, unknown_list[i].y, unknown_list[i].x ) );
		Rt( i, known - 1 ) = - DataWeightBg( CV_IMAGE_ELEM( alphamap, float, unknown_list[i].y, unknown_list[i].x ), CV_IMAGE_ELEM( confidencemap, float, unknown_list[i].y, unknown_list[i].x ) );
		row_sum += (float)Rt( i, known - 2 );
		row_sum += (float)Rt( i, known - 1 );

		//diagonal

		for(j=0;j<i;++j)
			row_sum += (float)Lu( i, j );

		Lu( i, i ) = -row_sum;		

		//printf("%d percent\r",(int)100*(i+1)/(int)unknown);
	}
	
	nonzero = 2*nonzero/*lower triangle*/ + unknown/*triangle*/;

	//
	dense1D<float> Ak(known);
	dense1D<float> Au(unknown);

	for(i=0;i<fg_list.size();++i)
		Ak[i] = 1;
	for(i=0;i<bg_list.size();++i)
		Ak[i + fg_list.size()] = 0;
	Ak[known - 2] = 1;
	Ak[known - 1] = 0;

	for(i=0;i<unknown_list.size();++i)
	{		
		Au[i] = CV_IMAGE_ELEM( alphamap, float, unknown_list[i].y, unknown_list[i].x );			
	}

	cvReleaseMat( &mat_3x3 );
	
	//solve Ax = b, where A = Lu, b = -R'Ak, x = Au			
	dense1D<float> bb(unknown);
	
	for(i=0;i<unknown;++i)
	{
		float sum = 0;
		for(j=0;j<known;++j)
			sum += ((float)Rt(i,j)*(float)Ak[j]);
		bb[i] = -sum;
	}

	//Note: only represent the upper triangle
	//Note: index starts from 1, which is different to the rules in C++
	A_value    = new double[nonzero];
	solution   = new double[unknown];
	b          = new double[unknown];
	A_rowIndex = new int[unknown+1];
	A_columns  = new int[nonzero];
	
	int count = 0;
	for(i=0;i<unknown;++i)
	{
		A_rowIndex[i] = count+1;
		A_columns[count] = i+1;
		A_value[count] = Lu( i, i );		//must encode diagonal element explicitly regardless of its value (zero or non-zero)

		++count;
		for(j=i+1;j<unknown;++j)
			if(Lu( i, j ))
			{
				A_columns[count] = j+1;
				A_value[count] = Lu( i, j );

				++count;
			}
	}
	A_rowIndex[unknown] = count+1;

	for(i=0;i<unknown;++i)
	{
		solution[i] = Au[i];
		b[i] = bb[i];
	}
}

double RobustMatting::Solve(bool __b)
{
	/* ==================================================================================
	    Initial Guess
	   ================================================================================== */
	//get the contour of foreground and background for sampling
	vector<CvPoint> fg_contour, bg_contour;
	GetContour( fgmask, fg_contour );
	GetContour( bgmask, bg_contour );	
	
	int i, j;
	vector<CvPoint> fg_samp, bg_samp;

	float conf, alpha;	
	for(i=0;i<colorImg->height;++i)
		for(j=0;j<colorImg->width;++j)
		{
			if(!CV_IMAGE_ELEM( unmask, uchar, i, j ))
				continue;

			//collect sample set
			CollectSampleSet( i, j, fg_samp, fg_contour );
			CollectSampleSet( i, j, bg_samp, bg_contour );

			//get confidence value and estimated alpha
			ConfidenceValue( i, j, fg_samp, bg_samp, conf, alpha );

			CV_IMAGE_ELEM( alphamap, float, i, j ) = alpha;
			CV_IMAGE_ELEM( confidencemap, float, i, j ) = conf;
		}
	
	cvNamedWindow("initial guess : alpha");
	cvShowImage( "initial guess : alpha", alphamap );
	cvNamedWindow("initial guess : confidence");
	cvShowImage( "initial guess : confidence", confidencemap );	
	cvWaitKey( 1 );

	/* ==================================================================================
	    Matte Optimization
	   ================================================================================== */
		
	double* A;
	double* x;
	double* b;
	int *rowIdx;
	int *col;
	int nonzero, length;
	vector<CvPoint> unknown_list;

	printf("building the sparse matrix to be solved...");
	BuildProblemMatrix( A, rowIdx, col, b, x, nonzero, length, unknown_list );

	//
	printf("\nsolving random walks by CG\n");
 	for(int iter = 0; iter<10; ++iter)
	{		
		solve_by_conjugate_gradient(A, rowIdx, col, b, x, nonzero, length, 1000);
		for(i=0;i<length;++i)
		{
			x[i] = __min( 1, __max( 0, x[i]));
		}
	}

	cvZero( alphamap );
	for(i=0;i<length;++i)
	{
		 CV_IMAGE_ELEM( alphamap, float, unknown_list[i].y, unknown_list[i].x ) = __min( 1.f, __max(0.f, (float)x[i]));
	}

	for(i=0;i<trimap->height;++i)
		for(j=0;j<trimap->width;++j)
		{			
			if( CV_IMAGE_ELEM( fgmask, uchar, i, j ) )							
				CV_IMAGE_ELEM( alphamap, float, i, j ) = 1;		
			else if( CV_IMAGE_ELEM( bgmask, uchar, i, j ) )
				CV_IMAGE_ELEM( alphamap, float, i, j ) = 0;		
		}

	cvNamedWindow("alpha");
	cvShowImage( "alpha", alphamap );
	cvWaitKey( 1 );

	printf("done!\n");

	delete [] A;
	delete [] x;
	delete [] b;
	delete [] rowIdx;
	delete [] col;

	return 1;
}

void RobustMatting::ConfidenceValue( int r, int c, const vector<CvPoint> &fg_samp, const vector<CvPoint> &bg_samp, float &conf, float &alpha )
{
	float Df, Db;
	Df = MinimumDistance( r, c, fg_samp );
	Db = MinimumDistance( r, c, bg_samp );

	CvPoint un, fg, bg;	
    int sampf, sampb;
	float conf_tmp, alpha_tmp;
	float conf_1, conf_2, conf_3, alpha_1, alpha_2, alpha_3;

	un = cvPoint( c, r );
	conf_1 = conf_2 = conf_3 = 0;
	alpha_1 = alpha_2 = alpha_3 = 0;
	
	for(sampf = 0;sampf < fg_samp.size();++sampf)
		for(sampb = 0;sampb < bg_samp.size();++sampb)
		{
			fg = fg_samp[sampf], bg = bg_samp[sampb];
			alpha_tmp =  EstimateAlpha( un, fg, bg );
			conf_tmp = expf( - ( DistanceRatioSquare( un, fg, bg, alpha_tmp ) * weight( un, fg, Df) * weight( un, bg, Db) ) / (sigma*sigma) );			

			if(conf_tmp>conf_1)
			{			
				conf_3 = conf_2;
				alpha_3 = alpha_2;
				
				conf_2 = conf_1;
				alpha_2 = alpha_1;

				conf_1 = conf_tmp;
				alpha_1 = alpha_tmp;
			}
			else if(conf_tmp>conf_2)
			{			
				conf_3 = conf_2;
				alpha_3 = alpha_2;

				conf_2 = conf_tmp;
				alpha_2 = alpha_tmp;
			}
			else if(conf_tmp>conf_3)
			{
				conf_3 = conf_tmp;
				alpha_3 = alpha_tmp;
			}
		};
	conf = (conf_1 + conf_2 + conf_3)/3.f;
	alpha = (alpha_1 + alpha_2 + alpha_3)/3.f;	
}

float RobustMatting::MinimumDistance( int r, int c, const vector<CvPoint> &samp_set )
{
	int i, mindist = INT_MAX;
	int d[3];
	int distance;
	for(i=0;i<samp_set.size();++i)
	{	
		d[0] = CV_IMAGE_ELEM( colorImg, uchar, samp_set[i].y, 3 * samp_set[i].x ) - CV_IMAGE_ELEM( colorImg, uchar, r, 3 * c );
		d[1] = CV_IMAGE_ELEM( colorImg, uchar, samp_set[i].y, 3 * samp_set[i].x + 1) - CV_IMAGE_ELEM( colorImg, uchar, r, 3 * c + 1);
		d[2] = CV_IMAGE_ELEM( colorImg, uchar, samp_set[i].y, 3 * samp_set[i].x + 2) - CV_IMAGE_ELEM( colorImg, uchar, r, 3 * c + 2);
		distance = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
		if( distance < mindist )
			mindist = distance;
	}

	return sqrtf((float)distance);
}

float RobustMatting::SmoothWeight( int r1, int c1, int r2, int c2, CvMat* mat_3x3 )
{
	int min_r = __min( r1, r2 ), min_c = __min( c1, c2 ), max_r = __max( r1, r2 ), max_c = __max( c1, c2);
	int start_r = __max( 0, max_r - 2 ), start_c = __max( 0, max_c - 2 );
	int end_r = __min( colorImg->height-3, min_r ), end_c = __min( colorImg->width-3, min_c );
		
	int r, c, i, j, color1, color2;
	float mean[3], sum[3], i_m[3], j_m[3];
	float one_nine = .111111f;//1.f/9.f;
	float w = 0;
	
	for(r=start_r;r<=end_r;++r)
		for(c=start_c;c<=end_c;++c)
		{
			memset( sum, 0, 3 * sizeof(float));
			cvZero( mat_3x3 );			

			CV_MAT_ELEM( *mat_3x3, float, 0, 0 ) += etta;
			CV_MAT_ELEM( *mat_3x3, float, 1, 1 ) += etta;
			CV_MAT_ELEM( *mat_3x3, float, 2, 2 ) += etta;

			for(i=0;i<3;++i)
				for(j=0;j<3;++j)
				{
					sum[0] += CV_IMAGE_ELEM( colorImg, uchar, r+i, 3 * (c+j) );
					sum[1] += CV_IMAGE_ELEM( colorImg, uchar, r+i, 3 * (c+j) + 1 );
					sum[2] += CV_IMAGE_ELEM( colorImg, uchar, r+i, 3 * (c+j) + 2 );
								
					for(color1=0;color1<3;++color1)
						for(color2=0;color2<3;++color2)
							CV_MAT_ELEM( *mat_3x3, float, color1, color2 )  += ( CV_IMAGE_ELEM( colorImg, uchar, r+i, 3 * (c+j) + color1 ) * CV_IMAGE_ELEM( colorImg, uchar, r+i, 3 * (c+j) + color2 ) );
				}		
						
			for(color1=0;color1<3;++color1)
				for(color2=0;j<color2;++color2)
					CV_MAT_ELEM( *mat_3x3, float, color1, color2 ) *= one_nine;
			
			mean[0] = one_nine * sum[0];
			mean[1] = one_nine * sum[1];
			mean[2] = one_nine * sum[2];

			CV_MAT_ELEM( *mat_3x3, float, 0, 0 ) -= (mean[0]*mean[0]);
			CV_MAT_ELEM( *mat_3x3, float, 0, 1 ) -= (mean[0]*mean[1]);
			CV_MAT_ELEM( *mat_3x3, float, 1, 0 ) -= (mean[0]*mean[1]);
			CV_MAT_ELEM( *mat_3x3, float, 0, 2 ) -= (mean[0]*mean[2]);
			CV_MAT_ELEM( *mat_3x3, float, 2, 0 ) -= (mean[0]*mean[2]);
			CV_MAT_ELEM( *mat_3x3, float, 1, 1 ) -= (mean[1]*mean[1]);
			CV_MAT_ELEM( *mat_3x3, float, 1, 2 ) -= (mean[1]*mean[2]);
			CV_MAT_ELEM( *mat_3x3, float, 2, 1 ) -= (mean[1]*mean[2]);
			CV_MAT_ELEM( *mat_3x3, float, 2, 2 ) -= (mean[2]*mean[2]);
			
			cvInvert( mat_3x3, mat_3x3 );

			i_m[0] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 ) - mean[0];
			i_m[1] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 + 1) - mean[1];
			i_m[2] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 + 2) - mean[2];
			
			j_m[0] = CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 ) - mean[0];
			j_m[1] = CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 + 1) - mean[1];
			j_m[2] = CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 + 2) - mean[2];

			sum[0] = i_m[0] * CV_MAT_ELEM( *mat_3x3, float, 0, 0 )
					 + i_m[1] * CV_MAT_ELEM( *mat_3x3, float, 0, 1 )
					 + i_m[2] * CV_MAT_ELEM( *mat_3x3, float, 0, 2 );
			sum[1] = i_m[0] * CV_MAT_ELEM( *mat_3x3, float, 1, 0 )
					 + i_m[1] * CV_MAT_ELEM( *mat_3x3, float, 1, 1 )
					 + i_m[2] * CV_MAT_ELEM( *mat_3x3, float, 1, 2 );
			sum[2] = i_m[0] * CV_MAT_ELEM( *mat_3x3, float, 2, 0 )
					 + i_m[1] * CV_MAT_ELEM( *mat_3x3, float, 2, 1 )
					 + i_m[2] * CV_MAT_ELEM( *mat_3x3, float, 2, 2 );

			w += (1 + sum[0] * j_m[0] + sum[1] * j_m[1] + sum[2] * j_m[2]);
		}

	return one_nine * w;
}

float RobustMatting::TestSmoothWeight( int r1, int c1, int r2, int c2, CvMat* mat_3x3 )
{
	int min_r = __min( r1, r2 ), min_c = __min( c1, c2 ), max_r = __max( r1, r2 ), max_c = __max( c1, c2);
	int start_r = __max( 0, max_r - 2 ), start_c = __max( 0, max_c - 2 );
	int end_r = __min( colorImg->height-3, min_r ), end_c = __min( colorImg->width-3, min_c );

	if(start_r>end_r || start_c>end_c)
		return 0;

	float c[3];
	c[0] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 ) - CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 );
	c[1] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 + 1 ) - CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 + 1 );
	c[2] = CV_IMAGE_ELEM( colorImg, uchar, r1, 3 * c1 + 2 ) - CV_IMAGE_ELEM( colorImg, uchar, r2, 3 * c2 + 2 );
	
	//printf("%f\n", expf(-(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])/25));
	return expf(-(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])/50);
}

#endif

FlashMatting::FlashMatting(IplImage *AmbientImg, IplImage *FlashImg, IplImage *Trimap)
{
	ambientImg = cvCloneImage( AmbientImg );
	flashImg   = cvCloneImage( FlashImg );	

	if(Trimap)
		trimap = cvCloneImage( Trimap );
	else
		trimap = NULL;    //generate the trimap automatically later

	Initialize();
	SetParameter();
}
FlashMatting::~FlashMatting()
{
	if(ambientImg)
		cvReleaseImage( &ambientImg );
	if(flashImg)
		cvReleaseImage( &flashImg );
	if(flashOnlyImg)
		cvReleaseImage( &flashOnlyImg );
	if(fgImg)
		cvReleaseImage( &fgImg );
	if(flashFgImg)
		cvReleaseImage( &flashFgImg );
	if(bgImg)
		cvReleaseImage( &bgImg );
	if(trimap)
		cvReleaseImage( &trimap );
	if(bgmask)
		cvReleaseImage( &bgmask );
	if(fgmask)
		cvReleaseImage( &fgmask );
	if(unmask)
		cvReleaseImage( &unmask );
	if(unsolvedmask)
		cvReleaseImage( &unsolvedmask );
	if(alphamap)
		cvReleaseImage( &alphamap );
}

void FlashMatting::SetParameter( int N, float sigma_, float sigma_i_square_, float sigma_ip_square_ )
{
	nearest = N;
	sigma   = sigma_;
	sigma_i_square = sigma_i_square_;
	sigma_ip_square = sigma_ip_square_;
}

void FlashMatting::Initialize()
{
	fgImg		  = cvCreateImage( cvGetSize( ambientImg ), 8, 3 );
	bgImg		  = cvCreateImage( cvGetSize( ambientImg ), 8, 3 );
	flashFgImg    = cvCreateImage( cvGetSize( ambientImg ), 8, 3 );
	flashOnlyImg  = cvCreateImage( cvGetSize( ambientImg ), 8, 3 );
	fgmask		  = cvCreateImage( cvGetSize( ambientImg ), 8, 1 );
	bgmask		  = cvCreateImage( cvGetSize( ambientImg ), 8, 1 );
	unmask		  = cvCreateImage( cvGetSize( ambientImg ), 8, 1 );
	unsolvedmask  = cvCreateImage( cvGetSize( ambientImg ), 8, 1 );
	alphamap      = cvCreateImage( cvGetSize( ambientImg ), 32, 1 );	

	cvZero( fgImg );
	cvZero( bgImg );
	cvZero( flashFgImg );
	cvZero( fgmask );
	cvZero( bgmask );
	cvZero( unmask );
	cvZero( alphamap );
	//cvZero( unsolvedmask );	

	//create flash-only image
	//IplImage* maxImg = cvCreateImage( cvGetSize( ambientImg ), 8, 3 );
	//cvMax( ambientImg, flashImg, maxImg );	//To make sure that there would be no negative value after substraction
	//cvSub( maxImg, ambientImg, flashOnlyImg );
	//cvReleaseImage( &maxImg );
	cvSub( flashImg, ambientImg, flashOnlyImg );

	if(!trimap)
		GenerateTrimap( flashOnlyImg, trimap );	


	int i, j, v;
	for(i=0;i<trimap->height;++i)
		for(j=0;j<trimap->width;++j)
		{
			v = CV_IMAGE_ELEM( trimap, uchar, i, j );
			if( v == BACKGROUND_VALUE )
				CV_IMAGE_ELEM( bgmask, uchar, i, j ) = 255;						
			else if( v == FOREGROUND_VALUE )
				CV_IMAGE_ELEM( fgmask, uchar, i, j ) = 255;		
			else
				CV_IMAGE_ELEM( unmask, uchar, i, j ) = 255;
		}

	cvSet( alphamap, cvScalarAll( 0 ), bgmask );
	cvSet( alphamap, cvScalarAll( 1 ), fgmask );
	cvCopyImage( unmask, unsolvedmask );
	
	cvCopy( ambientImg, fgImg, fgmask );
	cvCopy( ambientImg, bgImg, bgmask );

	cvCopy( flashOnlyImg, flashFgImg, fgmask );

	//cvNamedWindow("trimap");
	//cvNamedWindow("ambient");
	//cvNamedWindow("flash");
	//cvNamedWindow("flashOnly");
	//cvNamedWindow("fg");
	//cvNamedWindow("bg");
	//cvNamedWindow("flashFg");
	//cvShowImage("trimap", trimap );
	//cvShowImage("ambient", ambientImg );
	//cvShowImage("flash", flashImg );
	//cvShowImage("flashOnly", flashOnlyImg );
	//cvShowImage("fg", fgImg );
	//cvShowImage("bg", bgImg );
	//cvShowImage("flashFg", flashFgImg );
	//cvWaitKey( 0 );
}

void FlashMatting::GenerateTrimap( IplImage *FlashOnlyImg, IplImage *&Trimap, float level, float ratio )
{
	if(Trimap)
		cvReleaseImage( &Trimap );
	
	Trimap = cvCreateImage( cvGetSize(FlashOnlyImg ), 8, 1 );

    IplImage *grayImg = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );
    cvCvtColor( FlashOnlyImg, grayImg, CV_BGR2GRAY );

    //calculate histogram
    int i_bins = 128;
    int hist_size[] = {i_bins};
    float i_ranges[] = { 0, 255 };
    float *ranges[] = { i_ranges };

    CvHistogram *hist;
    hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY, ranges, 1 );

    IplImage *planes[] = { grayImg };

    cvCalcHist( planes, hist, 0, 0 );

    float histogram[128];
    for(int i=0;i<128;++i)
    {
            float weight;
            float weight_sum = 0, weighted_sum = 0;
            for(int j=max(0,i-10);j<=min(127,i+10);++j)
            {
#define VARIANCE 7.f
                    weight = expf( -(j-i)*(j-i)/ (2*VARIANCE));
                    weighted_sum += (cvQueryHistValue_1D( hist, j ) * weight);
                    weight_sum += weight;
            }
            histogram[i] = weighted_sum / weight_sum;
    }

    //find first local minimum
    int minBin = 0;
    for(int i=126;i>0;--i)
    {
            if(histogram[i]<histogram[i-1] && histogram[i]<histogram[i+1])
			{
                minBin = i;
				break;
			}			
    }

    int T = minBin*2+1;
	T *= level;
    //
	IplImage *highConfMask = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );
	IplImage *lowConfMask  = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );
	IplImage *mask         = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );
	IplImage *tmp          = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );
	IplImage *dilationImg    = cvCreateImage( cvGetSize( FlashOnlyImg ), 8, 1 );

	cvInRangeS( grayImg, cvScalar( T ), cvScalar(255), highConfMask );
	cvInRangeS( grayImg, cvScalar( ratio*T ), cvScalar(255), lowConfMask );

	cvZero( Trimap );
	while(cvCountNonZero(lowConfMask))
	{
		double min_val, max_val;
		CvPoint min_loc, max_loc;

		cvMinMaxLoc( lowConfMask, &min_val, &max_val, &min_loc, &max_loc );
		cvFloodFill( lowConfMask, max_loc, cvScalar( 128 ) );
		cvInRangeS( lowConfMask, cvScalar( 127 ), cvScalar(129), mask );
		
		cvZero( tmp );
		cvCopy( highConfMask, tmp, mask );
		if(cvCountNonZero( tmp ))
			cvSet( Trimap, cvScalar(255), mask );

		cvSet( lowConfMask, cvScalar(0), mask );
	}
	
	cvCopyImage( Trimap, tmp );	
	for(int iter = 0; iter < 20; ++iter)
	{	
		cvZero( dilationImg );
		for(int i=0;i<Trimap->height;++i)
			for(int j=0;j<Trimap->width;++j)	
				for(int di = max(0,i-1);di<=min(Trimap->height-1,i+1);++di)
					for(int dj = max(0,j-1);dj<=min(Trimap->width-1,j+1);++dj)
						CV_IMAGE_ELEM( dilationImg, uchar, i, j ) = max( CV_IMAGE_ELEM( dilationImg, uchar, i, j ) , CV_IMAGE_ELEM( tmp, uchar, di, dj ) );			
		cvCopyImage( dilationImg, tmp );
	}
		
	cvSub( dilationImg, Trimap, mask );

	cvSet( Trimap, cvScalar( 128 ), mask );

    cvReleaseImage( &grayImg );
	cvReleaseImage( &highConfMask );
	cvReleaseImage( &lowConfMask );
	cvReleaseHist( &hist );
	cvReleaseImage( &mask );
	cvReleaseImage( &tmp );
	cvReleaseImage( &dilationImg );
}

double FlashMatting::Solve(bool b = true)
{
	int p, r, c, i, j, k, iter;

	IplImage* shownImg       = cvCreateImage( cvGetSize(ambientImg), 8, 3 );
	IplImage* solveAgainMask = cvCreateImage( cvGetSize( unmask ), 8, 1 );

	vector<float>   fg_weight( FLASH_MAX_CLUS, 0 );
	vector<float>   bg_weight( FLASH_MAX_CLUS, 0 );
	vector<float>   flash_fg_weight( FLASH_MAX_CLUS, 0 );
	vector<CvMat *> fg_mean( FLASH_MAX_CLUS );
	vector<CvMat *> bg_mean( FLASH_MAX_CLUS );
	vector<CvMat *> flash_fg_mean( FLASH_MAX_CLUS );
	vector<CvMat *> inv_fg_cov( FLASH_MAX_CLUS );
	vector<CvMat *> inv_bg_cov( FLASH_MAX_CLUS );
	vector<CvMat *> inv_flash_fg_cov( FLASH_MAX_CLUS );
	for(i=0;i<FLASH_MAX_CLUS;i++)
	{
		fg_mean[i]		    = cvCreateMat(3,1,CV_32FC1);
		bg_mean[i]		    = cvCreateMat(3,1,CV_32FC1);
		flash_fg_mean[i]    = cvCreateMat(3,1,CV_32FC1);
		inv_fg_cov[i]		= cvCreateMat(3,3,CV_32FC1);
		inv_bg_cov[i]		= cvCreateMat(3,3,CV_32FC1);
		inv_flash_fg_cov[i] = cvCreateMat(3,3,CV_32FC1);
	}

	//for(int Iteration = 0 ; Iteration<3; ++Iteration )
	//{
	//	printf("\niteration %d:\n", Iteration);

	//	if(Iteration)
	//		cvCopy( unmask, solveAgainMask );

		int contour = 0;
		for(;;)
		{			
			printf("solving contour %d\r", contour++);

			vector<CvPoint> toSolveList;

	//		if(!Iteration)
				GetContour( unsolvedmask, toSolveList );
	//		else
	//			GetContour( solveAgainMask, toSolveList );
	//		
	//		//no unknown left
			if( !toSolveList.size() )
				break;

			cvCopyImage( ambientImg, shownImg );
			for(k=0;k<toSolveList.size();++k)
				cvCircle( shownImg, toSolveList[k], 1, cvScalarAll( 128 ) );
			cvNamedWindow( "points to solve" );
			cvShowImage( "points to solve", shownImg );
			cvWaitKey( 1 );

	//		//solve the points in the list one by one
			for( p = 0 ; p < toSolveList.size() ;++p )
			{	
				r = toSolveList[p].y, c = toSolveList[p].x;

				vector<pair<CvPoint,float> > fg_set, bg_set;
				CollectSampleSet( r, c, fg_set, bg_set );

				//get the gmm model using the neighbors of foreground and neighbors of background			
				GetGMMModel( r, c, fgImg, fg_set, fg_weight, fg_mean, inv_fg_cov );
				GetGMMModel( r, c, bgImg, bg_set, bg_weight, bg_mean, inv_bg_cov );
				GetGMMModel( r, c, flashFgImg, fg_set, flash_fg_weight, flash_fg_mean, inv_flash_fg_cov );
					
				int fgClus = 0, bgClus = 0, flashFgClus = 0;
				float L;
				float maxL = (float)-INT_MAX;
				for(i=0;i<FLASH_MAX_CLUS;i++)
					for(j=0;j<FLASH_MAX_CLUS;j++)
						for(k=0;k<FLASH_MAX_CLUS;k++)							
						{
							//initilize the alpha by the average of near points
		//					if(!Iteration)
								//InitializeAlpha( r, c, unsolvedmask );
		//					else
		//						InitializeAlpha( r, c, solveAgainMask );

							//initialze fg, bg, and flash-fg
							CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )   = CV_MAT_ELEM(*fg_mean[i], float, 0, 0 );
							CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 ) = CV_MAT_ELEM(*fg_mean[i], float, 1, 0 );
							CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 ) = CV_MAT_ELEM(*fg_mean[i], float, 2, 0 );
							CV_IMAGE_ELEM( bgImg, uchar, r, 3*c )   = CV_MAT_ELEM(*bg_mean[i], float, 0, 0 );
							CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ) = CV_MAT_ELEM(*bg_mean[i], float, 1, 0 );
							CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 ) = CV_MAT_ELEM(*bg_mean[i], float, 2, 0 );
							CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c )   = CV_MAT_ELEM(*flash_fg_mean[i], float, 0, 0 );
							CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+1 ) = CV_MAT_ELEM(*flash_fg_mean[i], float, 1, 0 );
							CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+2 ) = CV_MAT_ELEM(*flash_fg_mean[i], float, 2, 0 );
							
							for(iter=0;iter<3;++iter)
							{
								SolveAlpha( r, c );
								SolveBFF_( r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j], flash_fg_mean[k], inv_flash_fg_cov[k] );								
							}

							// largest likelihood, restore the index in fgClus, bgClus, flashFgClus 
							L = computeLikelihood( r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j], flash_fg_mean[k], inv_flash_fg_cov[k]);
							//L = computeLikelihood( r, c, fg_weight[i], fg_mean[i], inv_fg_cov[i], bg_weight[j], bg_mean[j], inv_bg_cov[j], flash_fg_weight[k], flash_fg_mean[k], inv_flash_fg_cov[k]);
							if(L>maxL)
							{
								maxL = L;
								fgClus = i;
								bgClus = j;
								flashFgClus = k;
							}
						}
								
				
	//			if(!Iteration)
					InitializeAlpha( r, c, unsolvedmask );
	//			else
	//				InitializeAlpha( r, c, solveAgainMask );

				for(iter=0;iter<5;++iter)
				{
					SolveBFF_( r, c, fg_mean[fgClus], inv_fg_cov[fgClus], bg_mean[bgClus], inv_bg_cov[bgClus], flash_fg_mean[flashFgClus], inv_flash_fg_cov[flashFgClus] );
					SolveAlpha( r, c );
				}
				//printf("%f\n", CV_IMAGE_ELEM(alphamap,float,r,c));

				//solved!
	//			if(!Iteration)
					CV_IMAGE_ELEM( unsolvedmask, uchar, r, c ) = 0;
	//			else
	//				CV_IMAGE_ELEM( solveAgainMask, uchar, r, c ) = 0;
			}
			//cvNamedWindow("fg");
			//cvShowImage("fg", fgImg );
			//cvNamedWindow("bg");
			//cvShowImage("bg", bgImg );
			//cvNamedWindow("flashFg");
			//cvShowImage("flashFg", flashFgImg );
			cvNamedWindow("alphamap");
			cvShowImage("alphamap", alphamap );
			cvWaitKey( 1 );
		}
	//}

	printf("\nDone!!\n");

	///////////////////////////

	cvReleaseImage( &shownImg );
	cvReleaseImage( &solveAgainMask );

	for(i=0;i<fg_mean.size();i++)
	{
		cvReleaseMat( &fg_mean[i] );
		cvReleaseMat( &bg_mean[i] );
		cvReleaseMat( &flash_fg_mean[i] );
		cvReleaseMat( &inv_fg_cov[i] );
		cvReleaseMat( &inv_bg_cov[i] );
		cvReleaseMat( &inv_flash_fg_cov[i] );
	}
	return 1;

}

void FlashMatting::GetContour( IplImage* img, vector<CvPoint> &contour )
{
	contour.clear();

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;

	cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

	for( ; contours != 0; contours = contours->h_next )
	{
		CvSeqReader reader;		
		cvStartReadSeq( contours, &reader, 0 );
		
		int i, count = contours->total;
		
		CvPoint pt;
		for( i = 0; i < count; i++ )
        {       
            CV_READ_SEQ_ELEM( pt, reader );
			contour.push_back( cvPoint( pt.x, pt.y ) );
        }
	}

	cvReleaseMemStorage( &storage );	
}

void FlashMatting::GetGMMModel( int r, int c, const IplImage *cImg, const vector<pair<CvPoint,float> > &point_set, vector<float> &weight, const vector<CvMat*> mean, const vector<CvMat*> inv_cov )
{	
	CvMat *work_mean = cvCreateMat(3,1,CV_32FC1);
	CvMat *cov = cvCreateMat(3,3,CV_32FC1);
	CvMat *work_inv_cov = cvCreateMat(3,3,CV_32FC1);
	CvMat *eigval = cvCreateMat(3,1,CV_32FC1);
	CvMat *eigvec = cvCreateMat(3,3,CV_32FC1);
	CvMat *cur_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *max_eigvec = cvCreateMat(3,1,CV_32FC1);
	CvMat *target_color = cvCreateMat(3,1,CV_32FC1);

	//// initializtion
	vector<pair<CvPoint,float> > clus_set[FLASH_MAX_CLUS];
	int nClus = 1;
	clus_set[0] = point_set;

	while(nClus<FLASH_MAX_CLUS)
	{
		// find the largest eigenvalue
		double max_eigval = 0;
		int max_idx = 0;
		for(int i=0;i<nClus;i++)
		{
			CalculateNonNormalizeCov(cImg,clus_set[i],work_mean,cov);

			// compute eigval, and eigvec
			cvSVD(cov, eigval, eigvec);
			if(cvmGet(eigval,0,0)>max_eigval)
			{
				cvGetCol(eigvec,max_eigvec,0);
				max_eigval = cvmGet(eigval,0,0);
				max_idx = i;
			}
		}

		// split
		float w;
		vector<pair<CvPoint,float> > new_clus_set[2];
		CalculateWeightMeanCov(cImg,clus_set[max_idx],w,work_mean,cov);
		double boundary = cvDotProduct(work_mean,max_eigvec);
		for(size_t i=0;i<clus_set[max_idx].size();i++)
		{
			for(int j=0;j<3;j++)
				cvmSet(cur_color,j,0,CV_IMAGE_ELEM( cImg, uchar, clus_set[max_idx][i].first.y, 3*clus_set[max_idx][i].first.x+j ));
			
			if(cvDotProduct(cur_color,max_eigvec)>boundary)
				new_clus_set[0].push_back(clus_set[max_idx][i]);
			else
				new_clus_set[1].push_back(clus_set[max_idx][i]);
		}

		clus_set[max_idx] = new_clus_set[0];
		clus_set[nClus] = new_clus_set[1];

		nClus += 1;
	}

	//// return all the mean and cov of fg
	float weight_sum, inv_weight_sum;
	weight_sum = 0;
	for(int i=0;i<nClus;i++)
	{
		CalculateWeightMeanCov(cImg,clus_set[i],weight[i],mean[i],cov);
		cvInvert(cov,inv_cov[i]);
		weight_sum += weight[i];
	}
	//normalize weight
	inv_weight_sum = 1.f / weight_sum;
	for(int i=0;i<nClus;i++)
		weight[i] *= inv_weight_sum;

	cvReleaseMat( &work_mean );
	cvReleaseMat( &cov );
	cvReleaseMat( &eigval );
	cvReleaseMat( &eigvec );
	cvReleaseMat( &cur_color );
	cvReleaseMat( &work_inv_cov );
	cvReleaseMat( &max_eigvec );
	cvReleaseMat( &target_color );
}

void FlashMatting::CalculateNonNormalizeCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov )
{
	int cur_r, cur_c;
	float cur_w, total_w=0;
	cvZero( mean );
	cvZero( cov );
	for(size_t j=0;j<clus_set.size();j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for(int h=0;h<3;h++)
		{
			CV_MAT_ELEM( *mean, float, h, 0 ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h ));
			for(int k=0;k<3;k++)
				CV_MAT_ELEM( *cov, float, h, k ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h )*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+k ));
		}

		total_w += clus_set[j].second;
	}
			
	float inv_total_w = 1.f/total_w;
	for(int h=0;h<3;h++)
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) -= (inv_total_w*CV_MAT_ELEM( *mean, float, h, 0 )*CV_MAT_ELEM( *mean, float, k, 0 ));

}

void FlashMatting::CalculateWeightMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, float &weight, CvMat* mean, CvMat* cov )
{
	int cur_r, cur_c;
	float cur_w, total_w=0;
	cvZero( mean );
	cvZero( cov );
	for(size_t j=0;j<clus_set.size();j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for(int h=0;h<3;h++)
		{
			CV_MAT_ELEM( *mean, float, h, 0 ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h ));
			for(int k=0;k<3;k++)
				CV_MAT_ELEM( *cov, float, h, k ) += (cur_w*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+h )*CV_IMAGE_ELEM( cImg, uchar, cur_r, 3*cur_c+k ));
		}

		total_w += clus_set[j].second;
	}
			
	float inv_total_w = 1.f/total_w;
	for(int h=0;h<3;h++)
	{
		CV_MAT_ELEM( *mean, float, h, 0 ) *= inv_total_w;
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) *= inv_total_w;
	}

	for(int h=0;h<3;h++)
		for(int k=0;k<3;k++)
			CV_MAT_ELEM( *cov, float, h, k ) -= (CV_MAT_ELEM( *mean, float, h, 0 )*CV_MAT_ELEM( *mean, float, k, 0 ));

	weight = total_w;
}

void FlashMatting::CollectSampleSet( int r, int c, vector<pair<CvPoint, float> > &fg_set, vector<pair<CvPoint, float> > &bg_set )
{	
	fg_set.clear(), bg_set.clear();	
#define UNSURE_DISTANCE 1	
	
	pair<CvPoint, float> sample;
	float dist_weight;
	float inv_2sigma_square = 1.f / (2*sigma*sigma);

	int dist = 1;
	while(fg_set.size() < nearest)
	{		
		if( r - dist >= 0 )
		{
			for(int z = max(0, c - dist); z<= min(ambientImg->width-1, c+dist); ++z )
			{ 
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r-dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r-dist, z) * CV_IMAGE_ELEM(alphamap, float, r-dist, z) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( r + dist < ambientImg->height)
		{			
			for(int z = max(0, c - dist); z<= min(ambientImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r+dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r+dist, z) * CV_IMAGE_ELEM(alphamap, float, r+dist, z) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( c - dist >= 0)
		{			
			for(int z = max(0, r - dist+ 1); z<= min(ambientImg->height-1, r+dist - 1 ); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c - dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c-dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c - dist) * CV_IMAGE_ELEM(alphamap, float, z, c - dist) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if( c + dist < ambientImg->width )
		{
			for(int z = max(0, r - dist + 1); z<= min(ambientImg->height-1, r + dist - 1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(fgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c + dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c+dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c + dist) * CV_IMAGE_ELEM(alphamap, float, z, c + dist) * dist_weight;

					fg_set.push_back( sample );
					if(fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		++dist;
	}

BG:
	int bg_unsure = 0;
	dist = 1;

	while(bg_set.size() < nearest)
	{
		dist_weight = expf( -(dist*dist)/(2*sigma*sigma) );
		if( r - dist >= 0 )
		{
			for(int z = max(0, c - dist); z<= min(ambientImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r-dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r-dist, z))
				{
					sample.first.x = z;
					sample.first.y = r-dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r-dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r-dist, z)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( r + dist < ambientImg->height)
		{
			for(int z = max(0, c - dist); z<= min(ambientImg->width-1, c+dist); ++z )
			{
				dist_weight = expf( -(dist*dist+(z-c)*(z-c)) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r+dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r+dist, z))
				{
					sample.first.x = z;
					sample.first.y = r+dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r+dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r+dist, z)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( c - dist >= 0)
		{
			for(int z = max(0, r - dist + 1); z<= min(ambientImg->height-1, r+dist-1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c-dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if( c + dist < ambientImg->width )
		{
			for(int z = max(0, r - dist + 1); z<= min(ambientImg->height-1, r+dist-1); ++z )
			{
				dist_weight = expf( -((z-r)*(z-r)+dist*dist) * inv_2sigma_square );

				if(CV_IMAGE_ELEM(bgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second  = dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
				else if( dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c+dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * dist_weight;

					bg_set.push_back( sample );
					if(bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		++dist;
	}

DONE:
	assert( fg_set.size() == nearest );
	assert( bg_set.size() == nearest );
}

void FlashMatting::InitializeAlpha( int r, int c, const IplImage* unSolvedMask )
{
	int i, j;
	int min_x, min_y, max_x, max_y;
#define WIN_SIZE 1


	min_x = max(0, c - WIN_SIZE);
	min_y = max(0, r - WIN_SIZE);
	max_x = min(ambientImg->width - 1, c + WIN_SIZE );
	max_y = min(ambientImg->height - 1, r + WIN_SIZE );

	int count = 0;
	float sum = 0;
	for( i = min_y; i<=max_y; ++i )
		for( j = min_x; j<=max_x; ++j )
		{
			if( !CV_IMAGE_ELEM(unSolvedMask, uchar, i, j) )
			{
				sum += CV_IMAGE_ELEM(alphamap, float, i, j);
				++count;
			}
		}
	
	CV_IMAGE_ELEM( alphamap, float, r, c ) = (count? sum / count : 0);	
}

inline void FlashMatting::SolveAlpha(int r, int c)
{
	CV_IMAGE_ELEM( alphamap, float, r, c ) =
	 (	 
		 sigma_ip_square *    
			( ((float)CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))   * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )     - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))
			+ ((float)CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c+1 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ))
			+ ((float)CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c+2 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) )
         + sigma_i_square * 
   		    ( (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c )   * (float)CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c )
			+ (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+1 ) * (float)CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c+1 )
			+ (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+2 ) * (float)CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c+2 ) )
	 ) / 
	 (
	     sigma_ip_square *
			( ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))   * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )     - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c ))
		    + ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 ))
	   	    + ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 ) - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) * ((float)CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 )   - (float)CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )) )
		+ sigma_i_square * 
   		    ( (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c )   * (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c )
			+ (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+1 ) * (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+1 )
			+ (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+2 ) * (float)CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+2 ) )
	 );

	CV_IMAGE_ELEM( alphamap, float, r, c ) = MAX( 0, MIN( 1, CV_IMAGE_ELEM( alphamap, float, r, c )));
}


void FlashMatting::SolveBFF_( int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov )
{
	CvMat *A	= cvCreateMat( 9, 9, CV_32FC1 );
	CvMat *x	= cvCreateMat( 9, 1, CV_32FC1 );
	CvMat *b	= cvCreateMat( 9, 1, CV_32FC1 );
	CvMat *I	= cvCreateMat( 3, 3, CV_32FC1 );
	CvMat *work_3x3 = cvCreateMat( 3, 3, CV_32FC1 );
	CvMat *work_3x1 = cvCreateMat( 3, 1, CV_32FC1 );
	
	float alpha = CV_IMAGE_ELEM( alphamap, float, r, c );	
	CvScalar  ambient_color = cvScalar( CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c ), CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c+1 ), CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c+2 ));
	CvScalar  flash_color   = cvScalar( CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c ), CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c+1 ), CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c+2 ));

	float inv_sigma_i_square  = 1.f / sigma_i_square;
	float inv_sigma_ip_square = 1.f / sigma_ip_square;

	cvZero( I );
	CV_MAT_ELEM( *I, float, 0, 0 ) = CV_MAT_ELEM( *I, float, 1, 1 ) = CV_MAT_ELEM( *I, float, 2, 2 ) = 1.f;

	////a
	cvZero( A );

	//
	cvCvtScale( I, work_3x3, alpha*alpha*inv_sigma_i_square );
	cvAdd( inv_fg_cov, work_3x3, work_3x3 );
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, i, j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	//
	cvCvtScale( I, work_3x3, alpha*(1-alpha)*inv_sigma_i_square);
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, i, 3+j ) = CV_MAT_ELEM( *A, float, 3+i, j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	//
	cvCvtScale( I, work_3x3, (1-alpha)*(1-alpha)*inv_sigma_i_square );
	cvAdd( inv_bg_cov, work_3x3, work_3x3 );
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, 3+i, 3+j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	//
	cvCvtScale( I, work_3x3, alpha*alpha*inv_sigma_ip_square );
	cvAdd( inv_flash_fg_cov, work_3x3, work_3x3 );
	for(int i=0;i<3;++i)
		for(int j=0;j<3;++j)
			CV_MAT_ELEM( *A, float, 6+i, 6+j ) = CV_MAT_ELEM( *work_3x3, float, i, j );

	////x
	cvZero( x );

	////b
	cvMatMul( inv_fg_cov, fg_mean, work_3x1 );
	for(int i=0;i<3;++i)
		CV_MAT_ELEM( *b, float, i, 0 ) = CV_MAT_ELEM( *work_3x1, float, i, 0 ) + (float)ambient_color.val[i]*alpha*inv_sigma_i_square;
	//
	cvMatMul( inv_bg_cov, bg_mean, work_3x1 );
	for(int i=0;i<3;++i)
		CV_MAT_ELEM( *b, float, 3+i, 0 ) = CV_MAT_ELEM( *work_3x1, float, i, 0 ) + (float)ambient_color.val[i]*(1-alpha)*inv_sigma_i_square;
	//
		////
	cvMatMul( inv_flash_fg_cov, flash_fg_mean, work_3x1 );
	for(int i=0;i<3;++i)
		CV_MAT_ELEM( *b, float, 6+i, 0 ) = CV_MAT_ELEM( *work_3x1, float, i, 0 ) + (float)flash_color.val[i]*alpha*inv_sigma_ip_square;

	//
	cvSolve( A, b, x );
	
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c )		 = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 0, 0 )));
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+1 )      = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 1, 0 )));
	CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+2 )      = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 2, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c )        = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 3, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+1 )      = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 4, 0 )));
	CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+2 )      = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 5, 0 )));
	CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c )   = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 6, 0 )));
	CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+1 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 7, 0 )));
	CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+2 ) = (uchar)MAX( 0, MIN( 255, CV_MAT_ELEM( *x, float, 8, 0 )));

	cvReleaseMat( &A );
	cvReleaseMat( &x );
	cvReleaseMat( &b );
	cvReleaseMat( &I );
	cvReleaseMat( &work_3x3 );
	cvReleaseMat( &work_3x1 );
}

float FlashMatting::computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov )
{
	float fgL, bgL, flashFgL, ambientL, flashL;
	int i;
	float alpha = CV_IMAGE_ELEM( alphamap, float, r, c );

	CvMat *work3x1 = cvCreateMat(3,1,CV_32FC1);
	CvMat *work1x3 = cvCreateMat(1,3,CV_32FC1);
	CvMat *work1x1 = cvCreateMat(1,1,CV_32FC1);
	CvMat *fg_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *bg_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *flash_fg_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *ambient_color = cvCreateMat(3,1,CV_32FC1);
	CvMat *flash_color = cvCreateMat(3,1,CV_32FC1);

	for(i=0;i<3;i++)
	{	
		CV_MAT_ELEM( *fg_color, float, i,0) = CV_IMAGE_ELEM( fgImg, uchar, r, 3*c+i );
		CV_MAT_ELEM( *bg_color, float, i,0) = CV_IMAGE_ELEM( bgImg, uchar, r, 3*c+i );
		CV_MAT_ELEM( *flash_fg_color, float, i,0) = CV_IMAGE_ELEM( flashFgImg, uchar, r, 3*c+i );
		CV_MAT_ELEM( *ambient_color, float, i,0) = CV_IMAGE_ELEM( ambientImg, uchar, r, 3*c+i );		
		CV_MAT_ELEM( *flash_color, float, i,0) = CV_IMAGE_ELEM( flashOnlyImg, uchar, r, 3*c+i );
	}

	// fgL
	cvSub(fg_color,fg_mean,work3x1);
	cvTranspose(work3x1,work1x3);
	cvMatMul(work1x3,inv_fg_cov,work1x3);
	cvMatMul(work1x3,work3x1,work1x1);
	fgL = -1.0f*CV_MAT_ELEM( *work1x1, float,0,0);

	// bgL
	cvSub(bg_color,bg_mean,work3x1);
	cvTranspose(work3x1,work1x3);
	cvMatMul(work1x3,inv_bg_cov,work1x3);
	cvMatMul(work1x3,work3x1,work1x1);
	bgL = -1.f*CV_MAT_ELEM( *work1x1, float,0,0);

	//flashFgL
	cvSub(flash_fg_color,flash_fg_mean,work3x1);
	cvTranspose(work3x1,work1x3);
	cvMatMul(work1x3,inv_flash_fg_cov,work1x3);
	cvMatMul(work1x3,work3x1,work1x1);
	flashFgL = -1.f*CV_MAT_ELEM( *work1x1, float,0,0);

	//ambientL
	cvAddWeighted(ambient_color, 1.0f, fg_color, -1.0f*alpha, 0.0f, work3x1 );
	cvAddWeighted(work3x1, 1.0f, bg_color, -1.0f*(1.0f-alpha), 0.0f, work3x1 );
	ambientL = -cvDotProduct( work3x1, work3x1 ) /sigma_i_square;

	//flashL
	cvAddWeighted(flash_color, 1.0f, flash_fg_color, -1.0f*alpha, 0.0f, work3x1 );	
	flashL = -cvDotProduct( work3x1, work3x1 ) /sigma_ip_square;

	cvReleaseMat( &work3x1 );
	cvReleaseMat( &work1x3 );
	cvReleaseMat( &work1x1 );
	cvReleaseMat( &fg_color );
	cvReleaseMat( &bg_color );
	cvReleaseMat( &flash_fg_color );
	cvReleaseMat( &ambient_color );
	cvReleaseMat( &flash_color );	

	return ambientL+flashL+fgL+bgL+flashFgL;
}

float FlashMatting::computeLikelihood( int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov, float flash_fg_weight, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov )
{
	return computeLikelihood( r, c, fg_mean, inv_fg_cov, bg_mean, inv_bg_cov, flash_fg_mean, inv_flash_fg_cov ) + logf(fg_weight)+logf(bg_weight)+logf(flash_fg_weight);
}



#ifdef POISSON_MATTING


PoissonMatting::PoissonMatting( IplImage* cImg, IplImage* tmap )
{
	colorImg = cvCloneImage( cImg );
	trimap   = cvCloneImage( tmap );	
	
	Initialize();
	SetParameter();
}

PoissonMatting::~PoissonMatting()
{
	if(colorImg)
		cvReleaseImage( &colorImg );
	if(grayImg)
		cvReleaseImage( &colorImg );
	if(FBImg)
		cvReleaseImage( &FBImg );
	if(divImg)
		cvReleaseImage( &divImg );
	if(trimap)
		cvReleaseImage( &trimap );
	if(alphamap)
		cvReleaseImage( &alphamap );	
	if(lastAlphamap)
		cvReleaseImage( &alphamap );	
	if(fgmask)
		cvReleaseImage( &fgmask );
	if(bgmask)
		cvReleaseImage( &bgmask );
	if(unmask)
		cvReleaseImage( &unmask );	
}

void PoissonMatting::Initialize()
{
	grayImg  = cvCreateImage( cvGetSize( trimap ), 8, 1 );	
	fgmask    = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	bgmask    = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	unmask   = cvCreateImage( cvGetSize( trimap ), 8, 1 );
	alphamap = cvCreateImage( cvGetSize( colorImg ), 32, 1 );	
	lastAlphamap = cvCreateImage( cvGetSize( colorImg ), 32, 1 );	
	FBImg    = cvCreateImage( cvGetSize( trimap ), 32, 1 );
	divImg    = cvCreateImage( cvGetSize( trimap ), 32, 1 );

	cvZero( fgmask );
	cvZero( bgmask );
	cvZero( unmask );
	cvZero( alphamap );

	cvCvtColor( colorImg, grayImg, CV_BGR2GRAY );

	int i, j, v;
	for(i=0;i<trimap->height-0;++i)
		for(j=0;j<trimap->width-0;++j)
		{
			v = CV_IMAGE_ELEM( trimap, uchar, i, j );
			if( v == BACKGROUND_VALUE )
				CV_IMAGE_ELEM( bgmask, uchar, i, j ) = 255;							
			else if( v == FOREGROUND_VALUE )			
				CV_IMAGE_ELEM( fgmask, uchar, i, j ) = 255;			
			else
				CV_IMAGE_ELEM( unmask, uchar, i, j ) = 255;			
		}
	
	//deal with the boundary problem, should not be unknown
	int tr, tc;
	for(i=0;i<trimap->height;++i)
		if( CV_IMAGE_ELEM( unmask, uchar, i, 0 ) )
		{
			if( nearestPoint( i, 0, tr, tc, fgmask ) < nearestPoint( i, 0, tr, tc, bgmask ) )
				CV_IMAGE_ELEM( fgmask, uchar, i, 0 ) = 255;
			else
				CV_IMAGE_ELEM( bgmask, uchar, i, 0 ) = 255;
			
			CV_IMAGE_ELEM( unmask, uchar, i, 0 ) = 0;
		}

	for(i=0;i<trimap->height;++i)
		if( CV_IMAGE_ELEM( unmask, uchar, i, trimap->width-1 ) )
		{
			if( nearestPoint( i, trimap->width-1, tr, tc, fgmask ) < nearestPoint( i, trimap->width-1, tr, tc, bgmask ) )
				CV_IMAGE_ELEM( bgmask, uchar, i, trimap->width-1 ) = 255;
			else
				CV_IMAGE_ELEM( fgmask, uchar, i, trimap->width-1 ) = 255;
			
			CV_IMAGE_ELEM( unmask, uchar, i, trimap->width-1 ) = 0;
		}

	for(j=0;j<trimap->width;++j)
		if( CV_IMAGE_ELEM( unmask, uchar, 0, j ) )
		{
			if( nearestPoint( 0, j, tr, tc, fgmask ) < nearestPoint( 0, j, tr, tc, bgmask ) )
				CV_IMAGE_ELEM( bgmask, uchar, 0, j ) = 255;
			else
				CV_IMAGE_ELEM( fgmask, uchar, 0, j ) = 255;
			
			CV_IMAGE_ELEM( unmask, uchar, 0, j ) = 0;
		}

	for(j=0;j<trimap->width;++j)
		if( CV_IMAGE_ELEM( unmask, uchar, trimap->height-1, j ) )
		{
			if( nearestPoint( trimap->height-1, j, tr, tc, fgmask ) < nearestPoint( trimap->height-1, j, tr, tc, bgmask ) )
				CV_IMAGE_ELEM( bgmask, uchar, trimap->height-1, j ) = 255;
			else
				CV_IMAGE_ELEM( fgmask, uchar, trimap->height-1, j ) = 255;
			
			CV_IMAGE_ELEM( unmask, uchar, trimap->height-1, j ) = 0;
		}

	//////

	cvSet( alphamap, cvScalarAll( 0 ), bgmask );
	cvSet( alphamap, cvScalarAll( 1 ), fgmask );
	cvCopyImage( alphamap, lastAlphamap );
}

void PoissonMatting::SetParameter()
{	
}

double PoissonMatting::Solve()
{
	InitialzeFB();

	for(int iter=0;iter<5;++iter)
	{	
		cvCopyImage( alphamap, lastAlphamap );


		ReconstructAlpha();
		


		printf("iteration : %d\n", iter );
		
		IplImage *tmp = cvCreateImage( cvGetSize( FBImg ), 32, 1 );
		cvNamedWindow( "alpha" );
		cvShowImage( "alpha", alphamap );
		cvWaitKey( 1 );
		cvReleaseImage( &tmp );

		cvSub( alphamap, lastAlphamap, lastAlphamap );
		CvScalar dif = cvAvg( alphamap, unmask );
		if(dif.val[0]<0.1)
			break;

		RefineFB();
	}
	
	printf("done!" );
	return 1;
}

void PoissonMatting::InitialzeFB()
{
	int r, c, fr, fc, br, bc;
	for(r=0;r<grayImg->height;++r)
		for(c=0;c<grayImg->width;++c)
		{
			nearestPoint( r, c, fr, fc, fgmask );
			nearestPoint( r, c, br, bc, bgmask );
			CV_IMAGE_ELEM( FBImg, float, r, c ) = CV_IMAGE_ELEM( grayImg, uchar, fr, fc ) - CV_IMAGE_ELEM( grayImg, uchar, br, bc );
		}

	cvSmooth( FBImg, FBImg, CV_GAUSSIAN, 5, 5, 0, 0 );

}

void PoissonMatting::ReconstructAlpha()
{
	getDiv();
	//

	int nRows, nCols;
	int mostNonZero;
	nRows = nCols = cvCountNonZero( unmask );
	mostNonZero = nRows * 5;	

	_INTEGER_t			*rowIndex = new _INTEGER_t[nRows+1];
	_INTEGER_t			*columns  = new _INTEGER_t[mostNonZero];
	_DOUBLE_PRECISION_t *values   = new _DOUBLE_PRECISION_t[mostNonZero];
	_DOUBLE_PRECISION_t *rhs      = new _DOUBLE_PRECISION_t[nCols];
	
	int r, c;

	int N = 0; // variable indexer
	map<int,int> mp;
	for (r = 1; r < unmask->height-1; r++) {
		for (c = 1; c < unmask->width-1; c++) {
			if(CV_IMAGE_ELEM(unmask, uchar, r, c))
			{
				int id = r*unmask->width+c;
				mp[id] = N;
				N++;
			}						
		}
	}

	
	//Note: index starts from 1, which is different to the rules in C++
	int n = 0;
	int index = 0;
	for(r=1;r<alphamap->height-1;++r)
		for(c=1;c<alphamap->width-1;++c)
		{
			if(CV_IMAGE_ELEM(unmask, uchar, r, c))
			{
				int id = r*unmask->width+c;
				rowIndex[n] = index+1;				
				
				//right hand side
				double bb = CV_IMAGE_ELEM(divImg, float, r, c);

				//
				if(CV_IMAGE_ELEM(unmask, uchar, r-1, c))
				{
					values[index] = 1.0f;
					columns[index] = mp[id-unmask->width]+1;
					index++;
				}
				else
				{
					bb -= CV_IMAGE_ELEM(alphamap, float, r-1, c);
				}				
				
				//
				if(CV_IMAGE_ELEM(unmask, uchar, r, c-1))
				{
					values[index] = 1.0f;
					columns[index] = mp[id-1]+1;
					index++;
				}
				else
				{
					bb -= CV_IMAGE_ELEM(alphamap, float, r, c-1);
				}
				
				//
				values[index] = -4.0f;
				columns[index] = mp[id]+1;
				index++;

				//
				if(CV_IMAGE_ELEM(unmask, uchar, r, c+1))
				{
					values[index] = 1.0f;
					columns[index] = mp[id+1]+1;
					index++;
				}
				else
				{
					bb -= CV_IMAGE_ELEM(alphamap, float, r, c+1);
				}

				//
				if(CV_IMAGE_ELEM(unmask, uchar, r+1, c))
				{
					values[index] = 1.0f;
					columns[index] = mp[id+unmask->width]+1;
					index++;
				}
				else
				{
					bb -= CV_IMAGE_ELEM(alphamap, float, r+1, c);
				}

				rhs[n] = bb;
				n++;
			}
		}

	assert(n == N);
	rowIndex[n] = index+1; // mark last CRS index
	
	int nNonZeros = index;

	/* Allocate storage for the solver handle and the right-hand side. */
	_DOUBLE_PRECISION_t *solValues = new _DOUBLE_PRECISION_t[nRows];
	_MKL_DSS_HANDLE_t handle;
	_INTEGER_t error;
	_CHARACTER_STR_t statIn[] = "determinant";
	_DOUBLE_PRECISION_t statOut[5];
	int opt = MKL_DSS_DEFAULTS;
	int non_sym = MKL_DSS_NON_SYMMETRIC;
	int type = MKL_DSS_INDEFINITE;
	//int type = MKL_DSS_POSITIVE_DEFINITE;

	/* --------------------- */
	/* Initialize the solver */
	/* --------------------- */
	error = dss_create(handle, opt );
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	/* ------------------------------------------- */
	/* Define the non-zero structure of the matrix */
	/* ------------------------------------------- */
	error = dss_define_structure(
	handle, non_sym, rowIndex, nRows, nCols,
	columns, nNonZeros );
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	/* ------------------ */
	/* Reorder the matrix */
	/* ------------------ */
	error = dss_reorder( handle, opt, 0);
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	/* ------------------ */
	/* Factor the matrix */
	/* ------------------ */
	error = dss_factor_real( handle, type, values );
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	/* ------------------------ */
	/* Get the solution vector */
	/* ------------------------ */
	int nRhs = 1;
	error = dss_solve_real( handle, opt, rhs, nRhs, solValues );
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	for (r = 1; r < unmask->height-1; r++) {
		for (c = 1; c < unmask->width-1; c++) {
			if(CV_IMAGE_ELEM(unmask, uchar, r, c))
			{
				int id = r*unmask->width+c;				
				CV_IMAGE_ELEM( alphamap, float, r, c ) = max( 0.f, min( 1.f, solValues[mp[id]] ));
			}						
		}
	}

	/* -------------------------- */
	/* Deallocate solver storage */
	/* -------------------------- */
	error = dss_delete( handle, opt );
	if ( error != MKL_DSS_SUCCESS ) goto printError;

	delete [] rowIndex;
	delete [] columns;
	delete [] values;
	delete [] rhs;
	delete [] solValues;

	return;

printError:
	printf("Solver returned error code %d\n", error);
	exit(1);
}

void PoissonMatting::RefineFB()
{
	IplImage *fgCand = cvCreateImage( cvGetSize(alphamap), 8, 1 );
	IplImage *bgCand = cvCreateImage( cvGetSize(alphamap), 8, 1 );

	cvInRangeS( alphamap, cvScalar( 0.95 ), cvScalar( 1.5 ), fgCand );
	cvInRangeS( alphamap, cvScalar( -0.5 ), cvScalar( 0.05 ), bgCand );
	int r, c, fr, fc, br, bc;
	for(r=0;r<alphamap->height;++r)
		for(c=0;c<alphamap->width;++c)
		{
			if(CV_IMAGE_ELEM(fgCand, uchar, r, c) && !CV_IMAGE_ELEM(fgmask, uchar, r, c))
			{
				nearestPoint( r, c, fr, fc, fgmask );
				if(abs(CV_IMAGE_ELEM(grayImg, uchar, r, c)-CV_IMAGE_ELEM(grayImg, uchar, fr, fc))>3)
					CV_IMAGE_ELEM(fgCand, uchar, r, c) = 0;
			}
			if(CV_IMAGE_ELEM(bgCand, uchar, r, c) && !CV_IMAGE_ELEM(bgmask, uchar, r, c))
			{
				nearestPoint( r, c, br, bc, bgmask );
				if(abs(CV_IMAGE_ELEM(grayImg, uchar, r, c)-CV_IMAGE_ELEM(grayImg, uchar, br, bc))>3)
					CV_IMAGE_ELEM(bgCand, uchar, r, c) = 0;
			}
		}

	///////
	for(r=0;r<alphamap->height;++r)
		for(c=0;c<alphamap->width;++c)
		{
			nearestPoint( r, c, fr, fc, fgCand );
			nearestPoint( r, c, br, bc, bgCand );
			CV_IMAGE_ELEM( FBImg, float, r, c ) = CV_IMAGE_ELEM( grayImg, uchar, fr, fc ) - CV_IMAGE_ELEM( grayImg, uchar, br, bc );
		}

	cvSmooth( FBImg, FBImg, CV_GAUSSIAN, 5, 5, 0, 0 );

	cvReleaseImage( &fgCand );
	cvReleaseImage( &bgCand );
}

void PoissonMatting::getDiv()
{
	IplImage *xGrad = cvCreateImage( cvGetSize( grayImg ), 32, 1 );
	IplImage *yGrad = cvCreateImage( cvGetSize( grayImg ), 32, 1 );
	
	cvSobel( grayImg, xGrad, 1, 0, 1 );
	cvSobel( grayImg, yGrad, 0, 1, 1 );

	IplImage *zeroMask = cvCreateImage( cvGetSize( grayImg ), 8, 1 );
	cvInRangeS( FBImg, cvScalar( -0.01 ), cvScalar( 0.01 ), zeroMask );
	cvSet( FBImg, cvScalar(0.01), zeroMask );

	cvDiv( xGrad, FBImg, xGrad );
	cvDiv( yGrad, FBImg, yGrad );

	cvSobel( xGrad, xGrad, 1, 0, 1 );
	cvSobel( yGrad, yGrad, 0, 1, 1 );

	cvAdd( xGrad, yGrad, divImg );

	cvReleaseImage( &xGrad );
	cvReleaseImage( &yGrad );
}

int PoissonMatting::nearestPoint(int r, int c, int &nr, int &nc, IplImage *mask)
{
	assert( cvCountNonZero(mask) );

	int dist = 0;
	int i, j;
	int min_dist = INT_MAX;
	for(i=0;i<mask->height;++i)
		for(j=0;j<mask->width;++j)
		{
			if(CV_IMAGE_ELEM(mask, uchar, i, j))
			{
				if((i-r)*(i-r)+(j-c)*(j-c)<min_dist)
				{
					min_dist = (i-r)*(i-r)+(j-c)*(j-c);
					nr = i;
					nc = j;
				}
			}
		}
	return min_dist;


	/*while(1)
	{
		if(r-dist>=0)
			for(j=max(0,c-dist);j<=min(mask->width-1,c+dist);++j )
				if(CV_IMAGE_ELEM(mask,uchar,r-dist,j))
				{
					nr = r-dist;
					nc = j;
					return dist;
				}
		
		if(r+dist<mask->height)
			for(j=max(0,c-dist);j<=min(mask->width-1,c+dist);++j )
				if(CV_IMAGE_ELEM(mask,uchar,r+dist,j))
				{
					nr = r+dist;
					nc = j;
					return dist;
				}

		if(c-dist>=0)
			for(i=max(0,r-dist);i<=min(mask->height-1,r+dist);++i )
				if(CV_IMAGE_ELEM(mask,uchar,i,c-dist))
				{
					nr = i;
					nc = c-dist;
					return dist;
				}
		
		if(c+dist<mask->width)
			for(i=max(0,r-dist);i<=min(mask->height-1,r+dist);++i )
				if(CV_IMAGE_ELEM(mask,uchar,i,c+dist))
				{
					nr = i;
					nc = c+dist;
					return dist;
				}

		dist++;
	}	*/
}

#endif