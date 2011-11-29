#pragma once

#include <vector>
using std::vector;
using std::pair;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <cv.h>

//#define ROBUST_MATTING 1

//#define max(a,b) (((a)>(b))?(a):(b))
//#define min(a,b) (((a)<(b))?(a):(b))

#define BACKGROUND_VALUE            0
#define FOREGROUND_VALUE          255

#define BAYESIAN_NUMBER_NEAREST   200
#define BAYESIAN_SIGMA            8.f
#define BAYESIAN_SIGMA_C          5.f
#define BAYESIAN_MAX_CLUS           2

#ifdef ROBUST_MATTING
#define ROBUST_SAMPLE_NUMBER      10
#define ROBUST_SAMPLE_DIST         2
#define ROBUST_SIGMA             .1f
#define ROBUST_GAMMA             5.f		//tune this one, the smaller the smoother
#define ROBUST_ETTA            1e-5f
#endif

#define FLASH_NUMBER_NEAREST      200
#define FLASH_SIGMA               8.f
#define FLASH_SIGMA_I_SQUARE     32.f
#define FLASH_SIGMA_IP_SQUARE    32.f
#define FLASH_MAX_CLUS              3

#ifdef POISSON_MATTING

#endif

class Matting
{
public:
	Matting(){ trimap = alphamap = NULL; }
	virtual double Solve(bool) = 0;
	IplImage *trimap;
	IplImage *alphamap;
};


//Implementation of	"A Bayesian Approach to Digital Matting"
class BayesianMatting : public Matting
{
public:
	/* ==================================================================================
		Constructors/ Destructors. 
	   ================================================================================== */

	//the input is the color image and the trimap image
	//the format is 3 channels + uchar and 1 channel + uchar respectively
	BayesianMatting( IplImage* cImg, IplImage* trimap );
	~BayesianMatting();
	
	void Initialize();

	//set parameter
	void SetParameter( int N = BAYESIAN_NUMBER_NEAREST, float sigma_ = BAYESIAN_SIGMA, float sigma_c = BAYESIAN_SIGMA_C );

	//solve the matting problem
	double Solve(bool);

	

private:	
	/* ==================================================================================
		Internal functions.
	   ================================================================================== */	
	//get the extreme outer contours of an image
	void GetContour( IplImage* img, vector<CvPoint> &contour );

	//initialize the alpha of one point using the mean of neighors and save the result in alphamap
	void InitializeAlpha( int r, int c, const IplImage* unSolvedMask );

	//used for clustering according the equation in paper "Color Quantization of Images"
	void CalculateNonNormalizeCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov );

	// calculate mean and cov of the given clus_set
	void CalculateMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov );
	
	// calculate weight, mean and cov of the given clus_set
	void CalculateWeightMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, float &weight, CvMat* mean, CvMat* cov );

	//get the foreground and backgroud gmm model at a given pixel
	void GetGMMModel( int r, int c, vector<float> &fg_weight, const vector<CvMat*> fg_mean, const vector<CvMat*> inv_fg_cov, vector<float> &bg_weight, const vector<CvMat*> bg_mean, const vector<CvMat*> inv_bg_cov );

	//collect the foreground/background sample set from the contour, called by GetGMMModel
	void CollectSampleSet( int r, int c, vector<pair<CvPoint, float> > &fg_set, vector<pair<CvPoint, float> > &bg_set );

	//solve Eq. (9) at pixel (r,c) according to the alpha in alphamap and save the result in fgImg and bgImg
	void SolveBF( int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov );

	//solve Eq. (10) at pixel (r,c) according to the foreground and background color in fgImg and bgImg, and save the result in alphamap
	inline void SolveAlpha( int r, int c );

	//compute total likelihood
	float computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov);

	float computeLikelihood(int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov);


	/* ==================================================================================
		Offline Variables.
	   ================================================================================== */
	int   nearest;	
	float sigma;
	float sigmac;
	
	/* ==================================================================================
		Online Variables.
	   ================================================================================== */
	
	IplImage *colorImg, *fgImg, *bgImg;
	IplImage *bgmask, *fgmask, *unmask, *unsolvedmask;
};


#ifdef ROBUST_MATTING

//Implementation of	"Optimized Color Sampling for Robust Matting"
class RobustMatting : public Matting
{
public:
	/* ==================================================================================
		Constructors/ Destructors. 
	   ================================================================================== */

	//the input is the color image and the trimap image
	//the format is 3 channels + uchar and 1 channel + uchar respectively
	RobustMatting( IplImage* cImg, IplImage* trimap );
	~RobustMatting();
	
	void Initialize();

	//set parameter
	void SetParameter( int sample_number = ROBUST_SAMPLE_NUMBER, int sample_dist = ROBUST_SAMPLE_DIST, float sigma = ROBUST_SIGMA, float gamma = ROBUST_GAMMA, float etta = ROBUST_ETTA );

	//solve the matting problem
	double Solve(bool);


private:	
	/* ==================================================================================
		Internal functions.
	   ================================================================================== */
	//get the extreme outer contours of an image
	void GetContour( IplImage* img, vector<CvPoint> &contour );
	
	//collect the sample set from the contour
	void CollectSampleSet( int r, int c, vector<CvPoint> &samp_set, const vector<CvPoint> &cand_set );
	
	//build problem matrix
	void BuildProblemMatrix( double *&A_value, int *&A_rowIndex, int *&A_columns, double *&b, double *&solution, int &nonzero, int &length, vector<CvPoint> &unknown_list );

	//get the confidence value and the estimated alpha value of an unknown pixel
	void ConfidenceValue( int r, int c, const vector<CvPoint> &fore_samp, const vector<CvPoint> &back_samp, float &conf, float &alpha );

	//minimum distance
	float MinimumDistance( int r, int c, const vector<CvPoint> &samp_set );	

	//smooth weight
	float SmoothWeight( int r1, int c1, int r2, int c2, CvMat* mat_3x3 );
	float TestSmoothWeight( int r1, int c1, int r2, int c2, CvMat* mat_3x3 );

	/* ==================================================================================
		Inline functions.
	   ================================================================================== */
	//estimate alpha given the color of unknown pixel, foreground pixel, and background pixel
	inline float EstimateAlpha( CvPoint un, CvPoint fg, CvPoint bg )
	{
		float alpha;
		int cb[3];
		int fb[3];		
		
		cb[0] = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x ) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x );
		cb[1] = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 1) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 1);
		cb[2] = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 2) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 2);
   		fb[0] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x ) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x );
		fb[1] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 1) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 1);
		fb[2] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 2) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 2);		
		alpha = (float)(cb[0]*fb[0] + cb[1]*fb[1] + cb[2]*fb[2]) / (float)(fb[0]*fb[0] + fb[1]*fb[1] + fb[2]*fb[2]);
		
		return __max( 0.f, __min( 1.f, alpha ) );
	};

	//square of equation (3)
	inline float DistanceRatioSquare( CvPoint un, CvPoint fg, CvPoint bg, float alpha )
	{
		float d[3];
		float fb[3];
		
		d[0]  = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x ) - 
			( alpha * CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x ) + (1-alpha) * CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x ) );
		d[1]  = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 1) - 
			( alpha * CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 1 ) + (1-alpha) * CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 1) );
		d[2]  = CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 2) - 
			( alpha * CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 2 ) + (1-alpha) * CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 2) );

		fb[0] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x ) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x );
		fb[1] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 1) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 1);
		fb[2] = CV_IMAGE_ELEM( colorImg, uchar, fg.y, 3 * fg.x + 2) - CV_IMAGE_ELEM( colorImg, uchar, bg.y, 3 * bg.x + 2);		
		
		return (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) / (fb[0]*fb[0] + fb[1]*fb[1] + fb[2]*fb[2]);
	}

	//equation (4), (5)
	inline float weight( CvPoint un, CvPoint g, float D )
	{
		float d[3];
		d[0] = CV_IMAGE_ELEM( colorImg, uchar, g.y, 3 * g.x ) - CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x );
		d[1] = CV_IMAGE_ELEM( colorImg, uchar, g.y, 3 * g.x + 1) - CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 1);
		d[2] = CV_IMAGE_ELEM( colorImg, uchar, g.y, 3 * g.x + 2) - CV_IMAGE_ELEM( colorImg, uchar, un.y, 3 * un.x + 2);
		return expf( -(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) / (D*D) );
	}

	//data weight, link to foreground	
	inline float DataWeightFg( float alpha, float confidence )
	{
		return gamma * (confidence * alpha + (1-confidence) * (alpha>0.5?1.f:0.f) );
	}

	//data weight, link to background	
	inline float DataWeightBg( float alpha, float confidence )
	{
		return gamma * (confidence * (1-alpha) + (1-confidence) * (alpha<0.5?1.f:0.f) );
	}

	//inline float InvSqrt (float x) {
	//	float xhalf = 0.5f*x;
	//	int i = *(int*)&x;
	//	i = 0x5f3759df - (i>>1);
	//	x = *(float*)&i;
	//	x = x*(1.5f - xhalf*x*x);
	//	return x;
	//}

	/* ==================================================================================
		Offline Variables.
	   ================================================================================== */
	int sample_number;
	int sample_dist;
	float sigma;
	float gamma;
	float etta;
	/* ==================================================================================
		Online Variables.
	   ================================================================================== */

	IplImage *colorImg;
	IplImage *bgmask, *fgmask, *unmask;
	IplImage *confidencemap;	
};

#endif

//Implementation of	"Flash Matting"
class FlashMatting : public Matting
{
public:
	/* ==================================================================================
		Constructors/ Destructors. 
	   ================================================================================== */

	//the input is the ambient image (no flash), flash image, and the trimap image is optional
	//the format is 3 channels + uchar 3 channels + uchar and 1 channel + uchar respectively
	FlashMatting( IplImage* ambientImg, IplImage* flashImg, IplImage* trimap = NULL );
	~FlashMatting();

	void Initialize();

	//set parameter
	void SetParameter( int N = FLASH_NUMBER_NEAREST, float sigma_ = FLASH_SIGMA, float sigma_i_square = FLASH_SIGMA_I_SQUARE, float sigma_ip_square = FLASH_SIGMA_IP_SQUARE );
	
	//automatically generate the trimap from flash/no-flash image pair
	static void GenerateTrimap( IplImage *FlashOnlyImg, IplImage *&Trimap, float level = 1.f, float ratio = 0.6f );

	//solve the matting problem
	double Solve(bool);
private:
	/* ==================================================================================
		Internal functions.
	   ================================================================================== */	
	//get the extreme outer contours of an image
	void GetContour( IplImage* img, vector<CvPoint> &contour );
	
	//get the model at a given pixel
	void GetGMMModel( int r, int c, const IplImage *cimg, const vector<pair<CvPoint,float> > &point_set, vector<float> &weight, const vector<CvMat*> mean, const vector<CvMat*> inv_cov );

	//used for clustering according the equation in paper "Color Quantization of Images"
	void CalculateNonNormalizeCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, CvMat* mean, CvMat* cov );
	
	// calculate weight, mean and cov of the given clus_set
	void CalculateWeightMeanCov( const IplImage *cImg, const vector<pair<CvPoint,float> > &clus_set, float &weight, CvMat* mean, CvMat* cov );

	//collect the foreground/background sample set from the contour, called by GetGMMModel
	void CollectSampleSet( int r, int c, vector<pair<CvPoint, float> > &point_set, vector<pair<CvPoint, float> > &bg_set );
	
	//initialize the alpha of one point using the mean of neighors and save the result in alphamap
	void InitializeAlpha( int r, int c, const IplImage* unSolvedMask );

	//solve Eq. (12) at pixel (r,c) according to the foreground, background color, and flash-only-fg color, and save the result in alphamap
	inline void SolveAlpha( int r, int c );

	//solve Eq. (13) at pixel (r,c) according to the alpha in alphamap and save the result in fgImg, flashFgimg, and bgImg
	void SolveBFF_( int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov );
	
	//compute total likelihood
	float computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov );
	float computeLikelihood( int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov, float flash_fg_weight, CvMat *flash_fg_mean, CvMat *inv_flash_fg_cov );
	/* ==================================================================================
		Offline Variables.
	   ================================================================================== */
	int   nearest;	
	float sigma;
	float sigma_i_square, sigma_ip_square;
	
	/* ==================================================================================
		Online Variables.
	   ================================================================================== */
	
	IplImage *ambientImg, *flashImg, *flashOnlyImg;
	IplImage *fgImg, *flashFgImg, *bgImg;
	IplImage *bgmask, *fgmask, *unmask, *unsolvedmask;
};

#ifdef POISSON_MATTING
//Implementation of	"A Bayesian Approach to Digital Matting"
class PoissonMatting : public Matting
{
public:
	/* ==================================================================================
		Constructors/ Destructors. 
	   ================================================================================== */

	//the input is the color image and the trimap image
	//the format is 3 channels + uchar and 1 channel + uchar respectively
	PoissonMatting( IplImage* cImg, IplImage* trimap );
	~PoissonMatting();
	
	void Initialize();

	//set parameter
	void SetParameter();

	//solve the matting problem
	double Solve();

	

private:	
	/* ==================================================================================
		Internal functions.
	   ================================================================================== */	
	void InitialzeFB();

	void ReconstructAlpha();

	void RefineFB();

	//return the square of nearest distance
	int nearestPoint( int r, int c, int &nr, int &nc, IplImage *mask );	

	void getDiv();

	/* ==================================================================================
		Offline Variables.
	   ================================================================================== */
	
	/* ==================================================================================
		Online Variables.
	   ================================================================================== */
	IplImage *colorImg, *grayImg;
	IplImage *FBImg, *divImg;
	IplImage *bgmask, *fgmask, *unmask;
	IplImage *lastAlphamap;
};

#endif