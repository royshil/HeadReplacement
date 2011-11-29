#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <ml.h>

using namespace cv;

#include <stdio.h>

static float T=0.6f;
static float T_step=0.1f;

void DrawStroke(int evt, int x, int y, int flags, void *param);
void RenderMsg( IplImage *display);
void initializeGUI(IplImage *img, IplImage *trimap, IplImage *flashOnlyImg=NULL);
void run();
void returnTrimap(IplImage *trimap );
void destructGUI();