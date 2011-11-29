#include "gui.h"
#include "matting.h"

// stroke type
#define _strokeu    2
#define _strokefg   1
#define _strokebg   0

// stroke color
int _strokeColor[] ={0, 255, 128};

// stroke
#define _swmin      1
#define _swmax      40
#define _swdefault  10

// state
#define _drawing    1
#define _idle       0

// global variable
static int stroketype = _strokefg;
static int strokewidth;
static int lastx, lasty, state;
IplImage *img_gui, *usr, *flashOnlyImg_gui;

void DrawStroke(int evt, int x, int y, int flags, void *param)
{
	CvScalar color = cvScalar( _strokeColor[stroketype]);
	if( evt == CV_EVENT_LBUTTONDOWN )
	{
		state = _drawing;
		lastx = x, lasty = y;
	}
	else if( evt == CV_EVENT_LBUTTONUP )
	{
		state = _idle;
		cvLine( usr, cvPoint(lastx, lasty), cvPoint(x,y), color, strokewidth);
	}
	else if( evt == CV_EVENT_MOUSEMOVE && state == _drawing )
	{
		cvLine( usr, cvPoint(lastx, lasty), cvPoint(x,y), color, strokewidth);
		lastx = x, lasty = y;
	}

	return;
}

void RenderMsg( IplImage *display)
{
	char msg[1000];
	CvFont font = cvFont( 1.0, 1);
	cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 1.5, 1.5, 0, 2, CV_AA);

	if( stroketype == _strokebg)
		cvPutText( display, "Stroke: +background", cvPoint( 0, display->height - 1), &font, CV_RGB( 255, 255, 255));
	else if( stroketype == _strokefg)
		cvPutText( display, "Stroke: +foreground", cvPoint( 0, display->height - 1), &font, CV_RGB( 255, 255, 255));
	else if( stroketype == _strokeu)
		cvPutText( display, "Stroke: eraser", cvPoint( 0, display->height - 1), &font, CV_RGB( 255, 255, 255));
	
	sprintf( msg, "brush size: %3d", strokewidth);
	cvPutText( display, msg, cvPoint( 0, 20), &font, CV_RGB( 0, 255, 0));

	return;
}

void initializeGUI(IplImage *img, IplImage *trimap, IplImage *flashOnlyImg)
{
	img_gui = cvCloneImage( img );
	usr = cvCloneImage( trimap );

	printf("----------------starting trimap drawing gui----------------\n");
	printf("press 'q' to leave\n");
	printf("press 'w' to change stroke type\n");
	printf("press 'a' to increase stroke width\n");
	printf("press 'd' to decrease stroke width\n");

	if(flashOnlyImg!=NULL)
	{
		flashOnlyImg_gui = cvCloneImage( flashOnlyImg );
		printf("press 'u' to increase threshold in auto-generating trimap\n");
		printf("press 'i' to decrease threshold in auto-generating trimap\n");	
	}
}

void run()
{
	int key;
	IplImage *mask = cvCreateImage( cvGetSize(img_gui), 8, 1 );
	IplImage *display = cvCreateImage( cvGetSize(img_gui), 8, 3 );

	IplImage *fgcolor = cvCloneImage( img_gui), *bgcolor = cvCloneImage( img_gui);
	cvSet( fgcolor, cvScalar( 255, 0, 0)), cvSet( bgcolor, cvScalar( 0, 255, 255));

	// gui
	int swmin = _swmin;
	int swmax = max( max( img_gui->width, img_gui->height) / 8, 1);
	int swstep = max( min( img_gui->width, img_gui->height) / 100, 1);
	int swdefault = min( max( min( img_gui->width, img_gui->height) / 32, swmin), swmax);

	strokewidth = swdefault;
	
	RenderMsg( display);
	cvNamedWindow( "working space" );
	cvNamedWindow( "trimap" );
	cvShowImage( "working space" , display );
	cvShowImage( "trimap" , usr); 
	
	cvSetMouseCallback( "working space", DrawStroke);
	cvSetMouseCallback( "trimap", DrawStroke);

	while(1)
	{
		key = cvWaitKey(5);
		if(key=='q')
			break;
		else if(key=='w')
		{
			stroketype++;
			stroketype = stroketype % 3;
			printf("%d\n", stroketype);
		}
		else if( key == 'd')	// decrease stroke width
			strokewidth = ( strokewidth - swstep < swmin) ? swmin : strokewidth - swstep;
		else if( key == 'a')	// increase stroke width
			strokewidth = ( strokewidth + swstep > swmax) ? swmax : strokewidth + swstep;
		else if( key == 'u' && flashOnlyImg_gui!=NULL )
		{
			T += T_step;
			FlashMatting::GenerateTrimap( flashOnlyImg_gui, usr, T);
		}
		else if( key == 'i' && flashOnlyImg_gui!=NULL )
		{
			T -= T_step;
			FlashMatting::GenerateTrimap( flashOnlyImg_gui, usr, T);
		}
			

		

		// display
		
		cvCopy( img_gui, display );
		cvCmpS( usr, _strokeColor[_strokebg], mask, CV_CMP_EQ);
		cvOr( img_gui, bgcolor, display, mask);
		cvCmpS( usr, _strokeColor[_strokefg], mask, CV_CMP_EQ);
		cvOr( img_gui, fgcolor, display, mask);
		cvConvertScale( display, display, 0.7);
		//cvCmpS( usr, _strokeColor[_strokeu], mask, CV_CMP_EQ);
		//cvCopy( img_gui, display, mask);
		
		RenderMsg( display);
		cvShowImage( "working space", display);
		cvShowImage( "trimap" , usr);
	}

	cvReleaseImage( &display );
	cvDestroyAllWindows();
}

void returnTrimap(IplImage *trimap )
{
	if(trimap)
		cvReleaseImage( &trimap );
	trimap = cvCloneImage( usr );
}

void destructGUI()
{
	cvReleaseImage( &img_gui );
	cvReleaseImage( &usr );

	printf("----------------leaving trimap drawing gui----------------\n");
}