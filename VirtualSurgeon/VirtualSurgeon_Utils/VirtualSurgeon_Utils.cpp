// VirtualSurgeon_Utils.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "VirtualSurgeon_Utils.h"

int curl_get(std::string& s, const std::string& _s = std::string(""));
int btm_wait_time = 0;

#include <tclap/CmdLine.h>
using namespace TCLAP;

#include "../libjson/libjson.h"

#include "neck_curve_points.h"

#include <fltk3/ask.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef WIN32
#include <Windows.h>
#endif

namespace VirtualSurgeon {

// Define the command line object.
CmdLine cmd("Command description message", ' ', "0.9");
bool cmd_initialized = false;
UnlabeledValueArg<string> filename_arg("filename","file to work on",false,"","string",cmd);
ValueArg<string> groundtruth_arg(string(),"groundtruth","ground truth file",false,"","string",cmd);

SwitchArg initialization_step_arg(	"a","initialization_step","Do initialization step?",cmd,false);
ValueArg<double> im_scale_by_arg(	"b","image-scale-by","Scale down image by factor",false,2.0,"float",cmd);
ValueArg<int> km_numc_arg(			"c","kmeans-num-centers","K-Means number of clusters",false,20,"int",cmd);
ValueArg<int> btm_wait_time_arg(	"d","wait-time","Time in msec to wait on debug pauses",false,1,"int",cmd);
ValueArg<int> relable_type_arg(		"e","relable-type","Type of relabeling in iterative procecess (0 = hist max, 1 = graph cut)",false,1,"int",cmd);
ValueArg<double> gb_freq_arg(		"f","gabor-freq","Gabor func frequency",false,0.15,"float",cmd);
ValueArg<double> gb_gamma_arg(		"g","gabor-gamma","Gabor func gamma",false,1.0,"float",cmd);
ValueArg<int> alpha_matt_dilate_arg("i","alpha-matt-dilate-size","Size in pixels to dilate mask for alpha matting",false,10,"int",cmd);
ValueArg<int> num_dog_arg(			"j","num-DoG","numer of DoG values in pixel-vector",false,2,"int",cmd);
SwitchArg do_kmeans_arg(			"k","do-kmeans","perform k-means in extract",cmd,false);
ValueArg<int> com_calc_type_arg(	"l","combine-type","Hist combine scores calc type (0=COREL, 1=CHISQR, 2=INTERSECT, 3=BAHAT.)",false,0,"int [0-3]",cmd);
ValueArg<int> km_numt_arg(			"m","kmeans-num-tries","K-Means number of tries",false,2,"int",cmd);
ValueArg<int> gb_nstds_arg(			"n","gabor-nstds","Gabor func number of std devs",false,3,"int",cmd);
SwitchArg consider_neighbors_arg(	"o","consider-neighbors","Consider pixel neighborhood in head extraction",cmd,false);
ValueArg<double> gb_phase_arg(		"p","gabor-phase","Gabor func phase",false,_PI/2.0,"float",cmd);
SwitchArg position_in_km_arg(		"q","position-in-km","Include position in K-Means?",cmd,false);
ValueArg<int> gc_iter_arg(			"r","grabcut-iterations","Number of grabcut iterations",false,1,"int",cmd);
ValueArg<double> gb_sig_arg(		"s","gabor-sigma","Gabor func sigma",false,1.0,"float",cmd);
ValueArg<double> com_thresh_arg(	"t","combine-threshold","Hist combine threshold",false,0.15,"float",cmd);
SwitchArg do_alphamatt_arg(			"u","do-alpha-matting","Do alpha matting?",cmd,false);
SwitchArg do_two_segments_arg(		"v","two-segments","Do only two-way segmentation?",cmd,false);
ValueArg<int> com_winsize_arg(		"w","combine-win-size","Hist combine window size",false,5,"int",cmd);
ValueArg<int> num_iters_arg(		"x","num-iters","Number of cut-backp iterations",false,3,"int",cmd);
ValueArg<int> com_add_type_arg(		"y","combine-add-type","Hist combine scores add type (0=L2, 1=MAX, 2=MAX+, 3=just +)",false,2,"int [0-3]",cmd);
ValueArg<int> gb_size_arg(			"z","gabor-size","Gabor filter bank size",false,2,"int",cmd);

SwitchArg two_back_kernels_arg(		string(),"do-two-back-kernels","use kernel for back and shoulders?",cmd,false);

SwitchArg use_hist_match_hs_arg(string(),"use-hist-match-hs","Use histogram matching over HS space for recoloring?",cmd,false);
SwitchArg use_hist_match_rgb_arg(string(),"use-hist-match-rgb","Use histogram matching over RGB space for recoloring?",cmd,false);
SwitchArg use_overlay_arg(string(),"use-overlay","Use overlay for recoloring?",cmd,false);
SwitchArg use_warp_rigid_arg(string(),"use-warp-rigid","Use rigid warp for neck warping?",cmd,false);
SwitchArg use_warp_affine_arg(string(),"use-warp-affine","Use affine warp for neck warping?",cmd,false);
SwitchArg use_double_warp_arg(string(),"use-double-warp","Use 2-way warping?",cmd,false);

ValueArg<float> hair_ellipse_size_mult_arg(string(),"hair-ellipse-size-factor","factor for in/decreasing size of hair ellipse",false,1.0,"float",cmd);

ValueArg<float> snake_w_dir_arg(string(),"snake-w-dir","snake dir weight",false,10.0,"float",cmd);
ValueArg<float> snake_w_shp_arg(string(),"snake-w-shp","snake shape weight",false,1.0,"float",cmd);
ValueArg<float> snake_w_ten_arg(string(),"snake-w-ten","snake tension weight",false,10.0,"float",cmd);
ValueArg<float> snake_w_edg_arg(string(),"snake-w-edg","snake edge weight",false,1000.0,"float",cmd);

ValueArg<int> poisson_blending_band_size_arg(string(),"poisson-blend-band","size of band for poisson blend",false,4.0,"int",cmd);

ValueArg<string> output_filename_arg(string(),"output","filename of output image",false,"","string",cmd);
SwitchArg no_gui_arg(string(),"no-gui","dont show any windows?",cmd,false);
SwitchArg female_arg(string(),"female","is the input image a female?",cmd,false);

SwitchArg blur_on_resize_arg(string(),"blur-on-resize","should gauss-blur on resizing?",cmd,false);
SwitchArg two_way_recoloring_arg(string(),"two-way-recoloring","should recolor two-way?",cmd,false);

void VirtualSurgeonParams::InitializeDefault() {
	VirtualSurgeonParams& params = *this;
	params.path_to_exe = "";
	params.gb_sig = 1.0;
	params.gb_freq = 0.15;
	params.gb_phase = _PI/2.0;
	params.gb_gamma = 1.0;
	params.gb_nstds = 3;
	params.gb_size = 2;
	params.km_numc = 20;
	params.com_winsize = 5;
	params.com_thresh = 0.25;
	params.com_add_type = 0;
	params.com_calc_type = 1;
	params.im_scale_by = 1;
	params.gc_iter = 0;
	params.km_numt = 1;
	params.doScore = false;
	params.relable_type = 1;
	params.doPositionInKM = false;
	params.doInitStep = false;
	params.num_cut_backp_iters = 2;
	params.do_alpha_matt = false;
	params.alpha_matt_dilate_size = 5;
	params.use_hist_match_hs = false;
	params.use_hist_match_rgb = false;
	params.use_overlay = false;
	params.use_warp_rigid = false;
	params.use_warp_affine = false;
	params.use_double_warp = false;	
	params.hair_ellipse_size_mult = 1;
	params.do_eq_hist = false;
	params.consider_pixel_neighbourhood = false;
	params.do_two_segments = false;
	params.do_kmeans = false;
	params.head_mask_size_mult = 1.0;
	params.num_DoG = 2;
	params.do_two_back_kernels = false;

	params.poisson_cloning_band_size = 4;

	params.snake_snap_edge_len_thresh = 200;
	params.snale_snap_total_width_coeff = 10;

	params.snake_snap_weight_consistency = 1.0;
	params.snake_snap_weight_direction = 10.0;
	params.snake_snap_weight_edge = 100.0;
	params.snake_snap_weight_tension = 10.0;


	params.no_gui = false;
	params.wait_time = 1;

	params.output_filename = "output.png";

	params.is_female = false;
	params.blur_on_resize = false;
	params.two_way_recolor = false;
	
#ifdef __APPLE__
	char path[1024];
	uint32_t size = sizeof(path);
	if (_NSGetExecutablePath(path, &size) == 0) {
		cout << "executable path is " << path<<endl;
		string path_s = path;
		this->path_to_exe = path_s.substr(0, path_s.rfind("/")) + "/";
#elif defined(WIN32)
	TCHAR szPath[MAX_PATH];
	
	if( GetModuleFileName( NULL, szPath, MAX_PATH ) )
	{
		string path_s = szPath;
		this->path_to_exe = path_s.substr(0, path_s.rfind("\\")) + "\\";
#endif
	} else {
		cerr << "can't get executable name" << endl;
		fltk3::alert("can't get executable name;");
	}
}

void VirtualSurgeonParams::ParseParams(int argc, const char** argv) {
	VirtualSurgeonParams& params = *this;

	InitializeDefault();

	try {  

	cmd.reset();

	// Parse the args.
	cmd.parse( argc, argv );

	// Get the value parsed by each arg. 
	params.filename = filename_arg.getValue();
	params.groundtruth = groundtruth_arg.getValue();
	params.gb_sig = gb_sig_arg.getValue();
	params.gb_freq = gb_freq_arg.getValue();
	params.gb_phase = gb_phase_arg.getValue();
	params.gb_gamma = gb_gamma_arg.getValue();
	params.gb_nstds = gb_nstds_arg.getValue();
	params.gb_size = gb_size_arg.getValue();
	params.km_numc = km_numc_arg.getValue();
	params.km_numt = km_numt_arg.getValue();
	params.com_thresh = com_thresh_arg.getValue();
	params.com_winsize = com_winsize_arg.getValue();
	params.com_add_type = com_add_type_arg.getValue();
	params.com_calc_type = com_calc_type_arg.getValue();
	params.im_scale_by = im_scale_by_arg.getValue();
	params.gc_iter = gc_iter_arg.getValue();
	//params.doScore = groundtruth_arg.isSet();
	params.relable_type = relable_type_arg.getValue();
	params.doPositionInKM = position_in_km_arg.getValue();
	params.doInitStep = initialization_step_arg.getValue();
	params.num_cut_backp_iters = num_iters_arg.getValue();
	params.wait_time = btm_wait_time_arg.getValue();
	params.do_alpha_matt = do_alphamatt_arg.getValue();
	params.alpha_matt_dilate_size = alpha_matt_dilate_arg.getValue();
	params.use_hist_match_hs = use_hist_match_hs_arg.getValue();
	params.use_hist_match_rgb = use_hist_match_rgb_arg.getValue();
	params.use_overlay = use_overlay_arg.getValue();
	params.use_warp_affine = use_warp_affine_arg.getValue();
	params.use_warp_rigid = use_warp_rigid_arg.getValue();
	params.use_double_warp = use_double_warp_arg.getValue();
	params.do_kmeans = do_kmeans_arg.getValue();
	params.do_two_back_kernels = two_back_kernels_arg.getValue();
	params.consider_pixel_neighbourhood = consider_neighbors_arg.getValue();
	params.num_DoG = num_dog_arg.getValue();
	params.do_two_segments = do_two_segments_arg.getValue();
	params.hair_ellipse_size_mult = hair_ellipse_size_mult_arg.getValue();

	params.snake_snap_weight_consistency = snake_w_shp_arg.getValue();
	params.snake_snap_weight_direction = snake_w_dir_arg.getValue();
	params.snake_snap_weight_edge = snake_w_edg_arg.getValue();
	params.snake_snap_weight_tension = snake_w_ten_arg.getValue();

	params.output_filename = output_filename_arg.getValue();
	params.no_gui = no_gui_arg.getValue();

	params.is_female = female_arg.getValue();
	params.blur_on_resize = blur_on_resize_arg.getValue();
	params.two_way_recolor = two_way_recoloring_arg.getValue();

	params.poisson_cloning_band_size = poisson_blending_band_size_arg.getValue();

	}catch (ArgException &e)  // catch any exceptions
	{ 
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl; scanf("press any key...\n"); exit(0);
	}
}

void VirtualSurgeonParams::PrintParams() {
	VirtualSurgeonParams& p = *this;

	cout<<"file to work on: "<<p.filename<<endl;
	cout<<"ground truth file: "<<p.groundtruth<<endl;
	cout<<"Gabor func frequency: "<<p.gb_freq<<endl;
	cout<<"Gabor func sigma: "<<p.gb_sig<<endl;
	cout<<"Gabor func phase: "<<p.gb_phase<<endl;
	cout<<"Gabor func gamma: "<<p.gb_gamma<<endl;
	cout<<"Gabor func number of std devs: "<<p.gb_nstds<<endl;
	cout<<"Gabor filter bank size: "<<p.gb_size;
	cout<<"K-Means number of clusters: "<<p.km_numc<<endl;
	cout<<"K-Means number of tries: "<<p.km_numt<<endl;
	cout<<"Hist combine window size: "<<p.com_winsize<<endl;
	cout<<"Hist combine threshold: "<<p.com_thresh<<endl;
	cout<<"Hist combine scores add type (0=L2, 1=MAX, 2=MAX+, 3=just +): "<<p.com_add_type<<endl;
	cout<<"Hist combine scores calc type (0=COREL, 1=CHISQR, 2=INTERSECT, 3=BAHAT.): "<<p.com_calc_type<<endl;
	cout<<"Scale down image by factor: "<<p.im_scale_by<<endl;
	cout<<"Number of grabcut iterations: "<<p.gc_iter<<endl;
	cout<<"Type of relabeling in iterative procecess (0 = hist max, 1 = graph cut): "<<p.relable_type<<endl;
	cout<<"Include position in K-Means?: "<<p.doPositionInKM<<endl;
	cout<<"Do initialization step?: "<<p.doInitStep<<endl;
	cout<<"Number of cut-backp iterations: "<<p.num_cut_backp_iters<<endl;
	cout<<"Time in msec to wait on debug pauses: "<<p.wait_time<<endl;
	cout<<"Do alpha matting?: "<<p.do_alpha_matt<<endl;
	cout<<"Size in pixels to dilate mask for alpha matting: "<<p.alpha_matt_dilate_size<<endl;
	cout<<"Use histogram matching over HS space for recoloring?: "<<p.use_hist_match_hs<<endl;
	cout<<"Use histogram matching over RGB space for recoloring?: "<<p.use_hist_match_rgb<<endl;
	cout<<"Use overlay for recoloring?: "<<p.use_overlay<<endl;
	cout<<"Use rigid warp for neck warping?: "<<p.use_warp_rigid<<endl;
	cout<<"Use affine warp for neck warping?: "<<p.use_warp_affine<<endl;
	cout<<"Use 2-way warping?: "<<p.use_double_warp<<endl;
	cout<<"Use only 2 segments for segmentation?"<<p.do_two_segments<<endl;
	cout<<"Poisson blending band size: " <<p.poisson_cloning_band_size<<endl;

	cout<<"Face:"<<endl<<"------"<<endl;
	cout<<"Left eye: "<<p.li.x<<","<<p.li.y<<endl;
	cout<<"Right eye: "<<p.ri.x<<","<<p.ri.y<<endl;
	cout<<"Pitch: "<<p.pitch<<endl;
	cout<<"Roll: "<<p.roll<<endl;
	cout<<"Yaw: "<<p.yaw<<endl;

	cout <<"Is female? "<<p.is_female<<endl;
	cout <<"blur on resize? "<<p.blur_on_resize<<endl;
	cout <<"two way recolor? "<<p.two_way_recolor<<endl;
}

void GetJSONPoint(Point& p, JSONNODE* n,int cols, int rows) {
	p.x = json_as_float((JSONNODE*) json_get(n,"x")) * cols / 100;
	p.y = json_as_float((JSONNODE*) json_get(n,"y")) * rows / 100;
}

void ParseJSON(JSONNODE *n, VirtualSurgeonFaceData& params, Mat& im) {
    if (n == NULL) {
        printf("Invalid JSON Node\n");
        return;
    }

	JSONNODE_ITERATOR i = json_begin(n);
    while (i != json_end(n)){
        if (*i == NULL){
            printf("Invalid JSON Node\n");
            return;
        }
 
        // recursively call ourselves to dig deeper into the tree
        if (json_type(*i) == JSON_ARRAY || json_type(*i) == JSON_NODE){
            ParseJSON(*i,params,im);
        }
 
        // get the node name and value as a string
        json_char *node_name = json_name(*i);
 
        // find out where to store the values
        if (strcmp(node_name, "eye_left") == 0) GetJSONPoint(params.li,*i,im.cols,im.rows);
        if (strcmp(node_name, "eye_right") == 0) GetJSONPoint(params.ri,*i,im.cols,im.rows);
        if (strcmp(node_name, "center") == 0) GetJSONPoint(params.center,*i,im.cols,im.rows);
		if (strcmp(node_name, "mouth_left") == 0) GetJSONPoint(params.mouth_left,*i,im.cols,im.rows);
		if (strcmp(node_name, "mouth_right") == 0) GetJSONPoint(params.mouth_right,*i,im.cols,im.rows);
		if (strcmp(node_name, "nose") == 0) GetJSONPoint(params.nose,*i,im.cols,im.rows);
        
        if (strcmp(node_name, "yaw") == 0) params.yaw = json_as_float(*i);
        if (strcmp(node_name, "roll") == 0) params.roll = json_as_float(*i);
        if (strcmp(node_name, "pitch") == 0) params.pitch = json_as_float(*i);
        

        // cleanup and increment the iterator
        json_free(node_name);
        ++i;
    }
}
	
void VirtualSurgeonFaceData::DetectEyes(Mat& frame){	
	String face_cascade_name = this->path_to_exe + "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = this->path_to_exe + "haarcascade_eye_tree_eyeglasses.xml";
	
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	if( !face_cascade.load( face_cascade_name ) ){ 
		cerr << "--(!)Error loading "<<face_cascade_name<<"\n";
		fltk3::alert("--(!)Error loading");
		return;
	}
	if( !eyes_cascade.load( eyes_cascade_name ) ){ cerr << "--(!)Error loading "<<eyes_cascade_name<<"\n"; return; };

	std::vector<Rect> faces;
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	
	li = ri = Point(-1,-1);
	
	for( int i = 0; i < faces.size(); i++ )
	{
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
		
		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;
		
		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		if (eyes.size() >= 2) {
			Point imcenter(frame.cols/2,frame.rows/2);
//			line(frame,imcenter,faces[i].tl() + eyes[0].tl(),Scalar(255,255,0),2);
			if (li.x != -1) {
				//already found eyes, favor the more "centric" eyes
				if (norm(faces[i].tl() + eyes[0].tl() - imcenter) > norm(li - imcenter)) { 
					//new found eyes are farther from the center
					continue;
				}
			}
			li = faces[i].tl() + (eyes[0].tl() + eyes[0].br())*0.5;
			ri = faces[i].tl() + (eyes[1].tl() + eyes[1].br())*0.5; //middle of rect - center of eye?
			if (li.x > ri.x) {//swap
				Point eye = li;
				li = ri;
				ri = eye;
			}
		}
//		for( int j = 0; j < eyes.size(); j++ )
//		{
//			Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 ); 
//			int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
//			circle( frame, center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
//		}
	} 
	if(faces.size() > 0 && li.x == -1) { //didn't find any pair of eyes...
		li = faces[0].tl();
		li.x += faces[0].width / 3;
		li.y += faces[0].height * 2 / 5;
		ri = li;
		ri.x += faces[0].width / 3;
	}
}

/**
This function goes to Face.com APIs (using CURL) to get the facial features of the face in the image.
It will take out the image URL from params.filename and fill in the ri,li,yaw,roll,pitch params, an also
load the image into the im argument
**/
bool VirtualSurgeonFaceData::FaceDotComDetection(Mat& im) {
	VirtualSurgeonFaceData& params = *this;

//	struct stat f__stat;

	cout << "Filename: " << params.filename << endl;

//	int indexofslash = params.filename.find_last_of("/");
//	int indexofqmark = params.filename.find_last_of("?");
	string img_filename = params.filename;
	//if(indexofslash >= 0) {
	//	img_filename = params.filename.substr(
	//		indexofslash+1,
	//		(indexofqmark>0)?indexofqmark-indexofslash-1:params.filename.length()-1
	//		);
	//}

	//cout << "Image filename " << img_filename << "...";

	////check if already downloaded before..
	//if(stat(img_filename.c_str(),&f__stat)!=0) {
	//	cout << "Download..."<<endl;
	//	if (!curl_get(params.filename,img_filename)) {
	//		cerr << "Cannot download picture file"<<endl;
	//		return;
	//	}
	//}

	//cout << " Found!" << endl;

	//if(img_filename.length() == 0) throw new Exception(-1,"Could not get normalized image filename","FaceDotComDetection","VirtualSurgeon_Utils.cpp",220);

	im = imread(img_filename);

	if(im.cols == 0 || im.rows == 0) {
		fltk3::alert("Can't read image");
		exit(0);
	}

	cout << "Image Read: " << endl;

	//check if already got Face.com detection before..
	string img_fn_txtext = img_filename.substr(0,img_filename.find_last_of(".")) + ".txt";
	cout << "Image text file: " << img_fn_txtext << endl;
	//if(stat(img_fn_txtext.c_str(),&f__stat)!=0) {
	//	//file with "txt" extension doesn't exist, so call Face.com API
	//	cout << "Get Face.com detection..."<<endl;
	//	string u("http://api.face.com/faces/detect.json?api_key=4bc70b36e32fc&api_secret=4bc70b36e3334&urls=");
	//	u += params.filename;

	//	curl_get(u,img_fn_txtext);
	//}

	//TODO: handle bad file format, or no detection
	{
		ifstream ifs(img_fn_txtext.c_str(),ifstream::in);
		if(ifs.is_open()) {
			string line;
			string _line;
			while(!ifs.eof())
			{
				ifs >> _line;
				line += _line;
			}
			ifs.close();

			cout << line;

			JSONNODE* n = json_parse(line.c_str());
			ParseJSON(n,params,im);
		} else {
			cerr << "can't open file " << img_fn_txtext << endl;
			return false;
		}

		/*
		stringstream ss(line);
		
		int p = line.find("eye_left\":{\"x\":");
		ss.seekg(p+strlen("eye_left\":{\"x\":"));
		double d;
		ss >> d;
		params.li.x = d*im.cols/100;
		p = ss.tellg();
		ss.seekg(p+5);
		ss >> d;
		params.li.y = d*im.rows/100;
		p = ss.tellg();
		ss.seekg(p+strlen("},\"eye_right\":{\"x\":"));
		ss >> d;
		params.ri.x = d*im.cols/100;
		p = ss.tellg();
		ss.seekg(p+5);
		ss >> d;
		params.ri.y = d*im.rows/100;

		ss.seekg(line.find("\"yaw\":") + strlen("\"yaw\":"));
		ss >> params.yaw;
		ss.seekg(line.find("\"roll\":") + strlen("\"roll\":"));
		ss >> params.roll;
		ss.seekg(line.find("\"pitch\":") + strlen("\"pitch\":"));
		ss >> params.pitch;
		*/
	}
	return true;
}

void VirtualSurgeonParams::face_grab_cut(Mat& orig, Mat& mask, int iters, int dilate_size) {
	Mat tmpMask(mask.rows,mask.cols,CV_8UC1,Scalar(GC_BGD));

	//create "buffer" zones for probably BG and prob. FG.
	{
		Mat __tmp(mask.rows,mask.cols,CV_8UC1,Scalar(0));
		dilate(mask,__tmp,Mat::ones(dilate_size,dilate_size,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT);	//probably background
		tmpMask.setTo(Scalar(GC_PR_BGD),__tmp);

		dilate(mask,__tmp,Mat::ones(dilate_size/2,dilate_size/2,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT); //probably foregroung
		tmpMask.setTo(Scalar(GC_PR_FGD),__tmp);

		erode(mask,__tmp,Mat::ones(dilate_size/3,dilate_size/3,CV_8UC1),Point(-1,-1),1,BORDER_REFLECT); // foreground
		tmpMask.setTo(Scalar(GC_FGD),__tmp);
	}

	//Mat(mask).copyTo(tmpMask);
	Mat bgdModel, fgdModel;

	if(!this->no_gui) {
		Mat _tmp;
		tmpMask.convertTo(_tmp,CV_32FC1);
		_tmp = tmpMask / 4.0f * 255.0f;
		imshow("tmp",_tmp);
		waitKey(wait_time);
	}

	cout << "Do grabcut... init... ";

	Rect mr;
	FindBoundingRect(mr,mask); //find_bounding_rect_of_mask(&((IplImage)mask));
	//initialize
	grabCut(
		orig,
		tmpMask,
		mr,
		bgdModel,
		fgdModel,
		1, GC_INIT_WITH_MASK);
#ifdef BTM_DEBUG
	cout << "run... ";
#endif
	for(int i=0;i<iters;i++) {
		//run one iteration
		grabCut(
			orig,
			tmpMask,
			mr,
			bgdModel,
			fgdModel,
			1);
#ifdef BTM_DEBUG
		cout << ".";
#endif
	}

	//cvShowImage("result",image);
	//cvCopy(__GCtmp,mask);
	//Mat(mask).setTo(Scalar(255),tmpMask);
	//cvSet(mask,cvScalar(255),&((IplImage)tmpMask));
	Mat __tm = tmpMask & GC_FGD;
	__tm.setTo(Scalar(255),__tm);
	__tm.copyTo(mask);

	if(!this->no_gui) {
		cout << "Done" << endl;
		imshow("tmp",mask);
		waitKey(wait_time);
	}

}

void FindBoundingRect(Rect& faceRect, const Mat& headMask) {
	Mat arow(1,headMask.cols,CV_32SC1);
	for(int i=0;i<headMask.cols;i++) {
		((int*)arow.data)[i] = i;
	}
	Mat allrows; repeat(arow,headMask.rows,1,allrows);
	Mat rows; allrows.copyTo(rows,(headMask > 0));
	rows.setTo(Scalar(headMask.cols/2),(headMask == 0));
	double minv,maxv;
	minMaxLoc(rows,&minv,&maxv);

	faceRect.x = minv;
	faceRect.width = maxv - minv;

	Mat acol(headMask.rows,1,CV_32SC1);
	for(int i=0;i<headMask.rows;i++) {
		((int*)acol.data)[i] = i;
	}
	Mat allcols; repeat(acol,1,headMask.cols,allcols);
	Mat cols; allcols.copyTo(cols,(headMask > 0));
	cols.setTo(Scalar(headMask.rows/2),(headMask == 0));
	minMaxLoc(cols,&minv,&maxv);

	faceRect.y = minv;
	faceRect.height = maxv - minv;
}

void takeBiggestCC(Mat& __mask, Mat bias) {
	if(bias.rows == 0 || bias.cols == 0) {
		bias.create(__mask.size(),CV_64FC1);
		bias.setTo(1);
	}

	vector<vector<Point> > contours;
	contours.clear();
	findContours(__mask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	//compute areas
	vector<double> areas(contours.size());
	for(int ai=0;ai<contours.size();ai++) {
		Mat _pts(contours[ai]);
		Scalar mp = mean(_pts);

		//bias score according to distance from center face
		areas[ai] = contourArea(Mat(contours[ai])) * bias.at<double>(mp.val[1],mp.val[0]);
	}

	//find largest connected component
	double max; Point maxLoc;
	minMaxLoc(Mat(areas),0,&max,0,&maxLoc);

	//draw back on mask
	__mask.setTo(Scalar(0)); //clear...
	drawContours(__mask,contours,maxLoc.y,Scalar(255),CV_FILLED);
}

void VirtualSurgeonParams::PoissonImageEditing(const Mat& back, const Mat& _backMask, const Mat& front, const Mat& _frontMask, bool doLaplacian, bool doBounding) {
	Mat backMask,frontMask;

	if(_backMask.type() != CV_8UC1)
		_backMask.convertTo(backMask,CV_8UC1,255.0);
	else
		backMask = _backMask;

	if(_frontMask.type() != CV_8UC1)
		_frontMask.convertTo(frontMask,CV_8UC1,255.0);
	else
		frontMask = _frontMask;

	assert(backMask.type() == frontMask.type());
	assert(backMask.type() == CV_8UC1);

	Mat _mask = backMask & frontMask;
	Rect bound; 
	if(doBounding) {
		FindBoundingRect(bound,_mask);
		bound.x = MAX(bound.x - this->poisson_cloning_band_size,0); 
		bound.y = MAX(bound.y - this->poisson_cloning_band_size,0); 
		bound.width = MIN(bound.width + this->poisson_cloning_band_size*2, _mask.cols - bound.x); 
		bound.height = MIN(bound.height + this->poisson_cloning_band_size*2, _mask.rows - bound.y);
	} else {
		bound.x = 0;
		bound.y = 0;
		bound.width = _mask.cols;
		bound.height = _mask.rows;
	}
	Mat mask; _mask(bound).copyTo(mask);


	assert(mask.type() == CV_8UC1);
	
	assert(back.channels() == 3);

	Mat back_64f; 
	//if(back.type() == CV_8UC3) {
	//	cout << "back.type() == CV_8UC3"<<endl;
		back(bound).convertTo(back_64f,CV_64FC3,1.0/255.0);
	//} else if(back.type() == CV_32FC3) {
	//	cout << "back.type() == CV_32FC3"<<endl;
	//	back(bound).convertTo(back_64f,CV_64FC3,1.0/255.0);
	//} else if(back.type() == CV_64FC3) {
	//	cout << "back.type() == CV_64FC3"<<endl;
	//	back(bound).copyTo(back_64f);
	//}

	if(!this->no_gui) {
		imshow("tmp",back_64f);
		waitKey(this->wait_time);
	}

	Mat front_64f; 
	//if(front.type() == CV_8UC3)
		front(bound).convertTo(front_64f,CV_64FC3,1.0/255.0);
	//else
	//	front(bound).convertTo(front_64f,CV_64FC3);

	if(!this->no_gui) {
		imshow("tmp",front_64f);
		waitKey(this->wait_time);
	}
	Laplacian(front_64f,front_64f,-1);
	
	if(!this->no_gui) {
		imshow("tmp",front_64f);
		waitKey(this->wait_time);
	}
	//Mat lap_back_64; Laplacian(back_64f,lap_back_64,-1);

	vector<Mat> v_b(3), v_f(3);//, v_b_lap(3); 
	split(back_64f,v_b); 
	split(front_64f,v_f);
	//split(lap_back_64,v_b_lap);
	
	int n = back_64f.cols;
	int m = back_64f.rows;
	int mn = m*n;
	
	if(!this->no_gui) {
		imshow("tmp",backMask);
		waitKey(this->wait_time);
		imshow("tmp",frontMask);
		waitKey(this->wait_time);
		imshow("tmp",mask);
		waitKey(this->wait_time);
	}

	#pragma omp parallel for schedule(dynamic)
	for(int color = 0; color < 3; color++) {
		gmm::row_matrix< gmm::rsvector<double> > M(mn,mn);
		OpenCV2ImageWrapper<uchar> maskImage(mask);
		ImageEditingUtils::matrixCreate(M, n, mn, maskImage);

		vector<double > solutionVectors(mn);
		vector<double> v_color(mn); 
		Mat v_c_mat(back.size(),CV_64FC1);

		v_c_mat.setTo(Scalar(0));
		solutionVectors.assign(mn,0.0);
		//v_color.assign(mn,0.0);

		if(doLaplacian)
			v_f[color].copyTo(v_c_mat,mask);

		//Mat tmp = v_b_lap[c] > v_c_mat;
		//v_b_lap[c].copyTo(v_c_mat,tmp);

		v_b[color].copyTo(v_c_mat,~mask);

		if(!this->no_gui) {
			imshow("tmp",v_c_mat);
			waitKey(this->wait_time);
		}

		v_c_mat.reshape(1,mn).copyTo(v_color);

		ImageEditingUtils::solveLinear(M,solutionVectors,v_color);

		#pragma omp critical
		{
			Mat(solutionVectors).reshape(1,m).convertTo(v_b[color],back.type(),255.0);
			cout <<"done with color " << color << endl;
		}

		if(!this->no_gui) {
			imshow("tmp",v_b[color]);
			waitKey(this->wait_time);
		}
	}

	Mat output; cv::merge(v_b,output);
	
	if(!this->no_gui) {
		imshow("tmp",output);
		waitKey(this->wait_time);
	}
	Mat _back_bound = back(bound);
	output.copyTo(_back_bound);
}

Mat_<Point2f>& VirtualSurgeonParams::LoadHeadCurve(Mat& im, Rect r) {
	if(m_curve.empty()) {
		string _filename = this->filename + ".estimated_neck.txt";
		std::ifstream head_curve_stream(_filename.c_str(), std::ifstream::in);
		m_curve = Mat_<Point2f>(135,1);
		if(head_curve_stream.is_open()) {
			float _x,_y; char delim;
			for(int i=0;i<135;i++) {	//X
				head_curve_stream >> _x >> delim; m_curve[i]->x = MIN(im.cols-1,MAX(0,_x - (r.width > 0 ? r.x : 0)));
			}
			for(int i=0;i<135;i++) {	//Y
				head_curve_stream >> _y >> delim; m_curve[i]->y = MIN(im.rows-1,MAX(0,_y - (r.height > 0 ? r.y : 0)));
			}
		} else {
			//throw "can't load head curve";
			//load "general" curve
			memcpy(m_curve.data, neck_curve_points_f, sizeof(float)*270);
			float scale_factor = norm(this->li - this->ri) * 0.3;
			//flip and scale
			transform(m_curve,m_curve,(Mat_<float>)(Mat_<float>(2,2) << scale_factor,0,0,-scale_factor));
			//translate
			Point2f face_center_curve = (m_curve(108) + m_curve(123)) * 0.5;
			Point2f face_center_input((this->li.x + this->ri.x) * 0.5,(this->li.y + this->ri.y) * 0.5);
//			cv::circle(im,face_center_input,10,Scalar(255),2);
//			cout << "face_center_curve " << face_center_curve << endl;
//			cout << "face_center_input " << face_center_input << endl;
			m_curve = m_curve + repeat((Mat_<Point2f>(1,1) << face_center_input),135,1);
			m_curve = m_curve - repeat((Mat_<Point2f>(1,1) << face_center_curve),135,1);
//			cv:circle(im,(m_curve(108) + m_curve(123)) * 0.5,10,Scalar(0,0,255),2);
		}
	}
	return m_curve;
}
	
	std::string VirtualSurgeonParams::GenerateUniqueID() {
		stringstream ss1; ss1 << cv::getTickCount();
		const std::string& _str = ss1.str();
		
		unsigned long hash = 5381;
		const char* str = _str.c_str();
		int c;
		
		while (c = *str++)
			hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
		
		stringstream ss; ss << hash << "_" << ss1.str();
		return ss.str();
    }
	

}//ns

#ifdef UTILS_MAIN
int main(int argc, char** argv) {
	/*Mat blue(200,200,CV_8UC3,Scalar(255,0,0));*/
	Mat blue(200,200,CV_8UC3); 
	//RNG& rng = theRNG();
	//rng.fill(blue,RNG::UNIFORM,Scalar(0,0,0),Scalar(256,256,256));
	Mat im = imread("../images/40406598_fd4e74d51c_d.jpg");
	im(Rect(200,200,200,200)).copyTo(blue);

	Mat red(200,200,CV_8UC3,Scalar(0,0,255));
	im = imread("../images/59600641_acd478ae71_d.jpg");
	im(Rect(200,200,200,200)).copyTo(red);

	Mat blumask = Mat::zeros(blue.size(),CV_8UC1);
	Mat redmask = Mat::zeros(blue.size(),CV_8UC1);
	circle(blumask,Point(100,75),50,Scalar(255),CV_FILLED);
	circle(redmask,Point(100,125),50,Scalar(255),CV_FILLED);

	namedWindow("tmp");
	imshow("tmp",blue);
	waitKey();
	imshow("tmp",red);
	waitKey();

	VirtualSurgeonParams p;
	p.InitializeDefaults();
	p.no_gui = false;
	p.PoissonImageEditing(red,redmask,blue,blumask);
}
#endif