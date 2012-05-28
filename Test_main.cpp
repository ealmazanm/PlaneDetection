#include <stdio.h>
#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2\calib3d\calib3d.hpp>
#include <ctype.h>
#include <algorithm>
#include <fstream>
#include "XnCppWrapper.h"

#define MAX_DEPTH 10000
#define NO_TRACKING	0
#define BUILD_MODEL	1
#define TRACK_MODEL	2

#define DISPLAY_FRAME 1
#define DISPLAY_BPROJ 0

using namespace std;
using namespace cv;


Point X1, X11, X2, X22;
ofstream outDebug("D:\\debug.txt", ios::out);

Point corner11, corner22, corner11Y, corner22Y;
int Xcoor,Ycoor;
bool clicked = false;

string PrintMode(int mode)
{
	const string NoTracking("NO_TRACKING");
	const string BuildModel("BUILD_MODEL");
	const string TrackModel("TRACK_MODEL");
	if(mode==NO_TRACKING) return(NoTracking);
	if(mode==BUILD_MODEL) return(BuildModel);
	if(mode==TRACK_MODEL) return(TrackModel);
}

void BuildDepthToGreyMapping(unsigned short *pMapping)
{
	for( int i=0; i<MAX_DEPTH; i++) // for visualisation
		pMapping[i] = 255.0*powf(1.0-((float)i/MAX_DEPTH),3.0); // entirely arbitrary
}

void ConvertXnDepthPixelToFrame(const XnDepthPixel *P, Mat M, unsigned short *pMapping)
{
	uchar *imagePtr = (uchar*)M.data;
	for (int y=0; y<XN_VGA_Y_RES*XN_VGA_X_RES; y++)
	{
		int charVal = pMapping[P[y]];
		imagePtr[3*y]   = charVal;
		imagePtr[3*y+1] = charVal;
		imagePtr[3*y+2] = charVal;
	}
}

void ConvertXnRGB24PixelToFrame(const XnRGB24Pixel *P, Mat M)
{
	uchar *imagePtr = (uchar*)M.data;
	for (int y=0; y<XN_VGA_Y_RES*XN_VGA_X_RES; y++)
	{
		imagePtr[3*y]   = P->nBlue;
		imagePtr[3*y+1] = P->nGreen;
		imagePtr[3*y+2] = P->nRed;
		P++;
	}
}

class Camera
{
public:
	XnUInt64 FocalLength;
	float ox, oy;
	XnDouble pSize;

	Point Camera::imageProject(Matx31d point)
	{
		Point P;
		P.x = (int) (ox + (point.val[0]/point.val[2])*(FocalLength/pSize));
		P.y = (int) (oy + (point.val[1]/point.val[2])*(FocalLength/pSize));
		return P;
	}
};

class Plane 
{
public:
	Matx31d parameters;	// (a,b,c) where z=ax+by+c
	Matx31d centroid;	// 3D centroid of data
	Matx31d normal;		// 3D normal (towards camera) of plane

	Rect	fitWindow;	// 2D image window containing plane
	double	residualVariance; // variance of residuals e=z-ax-by-c

	int	frontoPlaneFit(const XnDepthPixel*, Camera*, Rect, float, float);
	int	updatePlaneFit(const XnDepthPixel*, const XnRGB24Pixel* colors, Camera*, Rect, Mat& weighFrameDepth, Mat& wighFrameColor, bool flag);
};



void getCoordinates_onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Xcoor = x;
		Ycoor = y;
		clicked = true;
		
//		XnDepthPixel* depthMap = (XnDepthPixel*)param;
//		int xCoor, yCoor;
//		XnFloat depth = depthMap[yCoor * XN_VGA_X_RES + xCoor];
//		if (depth == 0)
//		{
//			cout << "Got noise." << endl;
////			recoverFromNoiseSinglePoint(xCoor, yCoor, depthMap);
//		}
//		else
//		{
//
//		}
	}
}

int main()
{
	// VARIABLES -------------------------------------------------------------------------

	// Generator variables
	xn::Context context;
	XnStatus nRetVal;
	xn::ImageGenerator imageGenerator;
	xn::DepthGenerator depthGenerator;

	// Control variables
	bool bContinueLoop = true;
	int err, trackingMode = NO_TRACKING;
	int displayMode = DISPLAY_FRAME;

	// Intrinsic variables
	Camera kinectCamera;

	// Image Variables
	unsigned short Mapping[MAX_DEPTH];
	Mat Frame(XN_VGA_Y_RES, XN_VGA_X_RES,CV_8UC3);
	Mat Depth(XN_VGA_Y_RES, XN_VGA_X_RES,CV_8UC3);
	

	// Tracking variables
	Point origin(3*XN_VGA_X_RES/8,3*XN_VGA_Y_RES/8), corner(5*XN_VGA_X_RES/8,5*XN_VGA_Y_RES/8);
	Rect selectWindow;
	RotatedRect trackBox;

	// Plane variables 
	Plane myPlane;
	float minDepth=500, maxDepth=2000;

	// INITIALISE ------------------------------------------------------------------------

	// Initialise sensors
	nRetVal = context.Init();

	context.OpenFileRecording("D:\\TwoKinectsRecording\\cam111.oni");
	context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);
	context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);

//	nRetVal = imageGenerator.Create(context);
//	nRetVal = depthGenerator.Create(context);

	nRetVal = context.StartGeneratingAll();

	if (depthGenerator.IsCapabilitySupported("AlternativeViewPoint"))
		depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);


	// Initialise Camera with Intrinsics
	depthGenerator.GetRealProperty ("ZPPS", kinectCamera.pSize);
	kinectCamera.pSize *= 2.0;
	depthGenerator.GetIntProperty ("ZPD",kinectCamera.FocalLength);
	kinectCamera.ox = XN_VGA_X_RES/2;
	kinectCamera.oy = XN_VGA_Y_RES/2;

	// DISPLAY
	BuildDepthToGreyMapping(Mapping);
	cvNamedWindow("Depth Frame", 1);
	cvNamedWindow("Image Frame", 1);
	cvNamedWindow("Weigh Frame",1);

	// Set selection window
	selectWindow.x = origin.x;
    selectWindow.y = origin.y;
    selectWindow.width = corner.x-origin.x;
    selectWindow.height = corner.y-origin.y;
	int c = 0;
	bool flag = false;
	double FACTOR = kinectCamera.pSize/kinectCamera.FocalLength;
	// MAIN LOOP -------------------------------------------------------------------------
	while (bContinueLoop)
	{
//		if (c++ == 90)
//			trackingMode = BUILD_MODEL;

		Mat WeighImg_I(XN_VGA_Y_RES, XN_VGA_X_RES,CV_32FC1);
		Mat WeighImg_II(XN_VGA_Y_RES, XN_VGA_X_RES,CV_32FC1);
		// Recover data from sensors
		context.WaitAndUpdateAll();
		const XnRGB24Pixel* pImageMap = imageGenerator.GetRGB24ImageMap();
		const XnDepthPixel* pDepthMap = depthGenerator.GetDepthMap();

		// PRODUCE DEPTH MAP: Copies and converts depth image to greyscale (For display only)
		ConvertXnDepthPixelToFrame(pDepthMap,Depth,Mapping);

		// Copy RGB Generator data into Mat Frame (For display only)
		ConvertXnRGB24PixelToFrame(pImageMap,Frame);
 
//		cout << "[Mode=" << PrintMode(trackingMode) << "] ";

		if( trackingMode!=NO_TRACKING )
        {
			// Build model
			if( trackingMode==BUILD_MODEL )
			{
				err = myPlane.frontoPlaneFit( pDepthMap, &kinectCamera, selectWindow, minDepth,maxDepth );
	
rectangle(Frame, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(0,0,255), 1,1,0);
rectangle(WeighImg_I, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(0,0,255), 1,1,0);			
rectangle(WeighImg_II, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(0,0,255), 1,1,0);			

				if((err!=0)||(myPlane.fitWindow.area()<=1))
				{
					trackingMode=NO_TRACKING;
					cout << "No initial plane found\n";
				}
				else trackingMode=TRACK_MODEL;
			}
			
			// Track Model
			else
			{
				err = myPlane.updatePlaneFit( pDepthMap, pImageMap, &kinectCamera, myPlane.fitWindow, WeighImg_I, WeighImg_II, flag);

	rectangle(Frame, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(0,0,255), 1,1,0);			
	circle(Frame, Point(myPlane.fitWindow.x, myPlane.fitWindow.y),2, Scalar(255,0,0));
	circle(Frame, Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),2, Scalar(43,107,255));
	rectangle(WeighImg_I, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(255,255,255), 1,1,0);			
	rectangle(WeighImg_II, Point(myPlane.fitWindow.x, myPlane.fitWindow.y), Point(myPlane.fitWindow.x+myPlane.fitWindow.width, myPlane.fitWindow.y+myPlane.fitWindow.height),Scalar(255,255,255), 1,1,0);			
				if((err!=0)||(myPlane.fitWindow.area()<=1))
				{
					cout << " Error=" << err << ", Area=" << myPlane.fitWindow.area() << ", ";
					cout << "[" << myPlane.fitWindow.width << "X" << myPlane.fitWindow.height << "] at (" << myPlane.fitWindow.x << "X" << myPlane.fitWindow.y << ") ";
					trackingMode=NO_TRACKING;
					myPlane.residualVariance = 900;
					cout << "Tracked plane lost\n";
				}
				else
					cout << " StdDev=" << sqrt(myPlane.residualVariance);
			}

			// Image DISPLAY
			rectangle( Frame, origin, corner, Scalar(255,0,0), 1,1,0);
			Point centroid = kinectCamera.imageProject( myPlane.centroid );
			Point normal   = kinectCamera.imageProject( myPlane.centroid + 300*myPlane.normal );
			line(Frame, centroid, normal, Scalar(0,0,255), 2, 3, 0);

			// Depth DISPLAY
			Point P1(myPlane.fitWindow.x,myPlane.fitWindow.y);
			Point P2(myPlane.fitWindow.x+myPlane.fitWindow.width,myPlane.fitWindow.y+myPlane.fitWindow.height);

			rectangle( Depth, P1, P2, Scalar(0,255,255), 1,1,0);
		}
		
		Matx31f centered (0, 0, 700);
		Matx31f pointX(100, 0, 700);
		Matx31f pointY(0, 100, 700);
		Matx31f pointZ(0, 0, 800);
		Point centroid2D = kinectCamera.imageProject(centered);
		Point point2DX = kinectCamera.imageProject(pointX);
		Point point2DY = kinectCamera.imageProject(pointY);
		Point point2DZ = kinectCamera.imageProject(pointZ);
		line(Frame, centroid2D, point2DX, Scalar(255,0,0), 1,1,0);
		line(Frame, centroid2D, point2DY, Scalar(0,0,255), 1, 1, 0);
		line(Frame, centroid2D, point2DZ, Scalar(0,255,0), 1, 1, 0);

		circle(Frame, corner11, 2, Scalar::all(0), 3);
		circle(Frame, corner22, 2, Scalar::all(255), 3);

		circle(Frame, corner11Y, 2, Scalar::all(0), 3);
		circle(Frame, corner22Y, 2, Scalar::all(255), 3);

		// Display
		rectangle( Frame, origin, corner, Scalar(255,0,0), 1,1,0);
		//circle(Frame, X1, 2, Scalar::all(0), 3);
		//circle(Frame, X2, 2, Scalar::all(255), 3);
		//circle(Frame, X11, 2, Scalar::all(0), 3);
		//circle(Frame, X22, 2, Scalar::all(255), 3);
//		rectangle( WeighImg, origin, corner, Scalar(255,0,0), 1,1,0);
	//	flip(Frame, Frame,1);
		imshow("Image Frame", Frame);
		setMouseCallback("Image Frame", getCoordinates_onMouse, (XnDepthPixel*)pDepthMap);

		imshow("Depth Frame", Depth);
		imshow("Weigh Frame I", WeighImg_I);
		imshow("Weigh Frame II", WeighImg_II);
		flag = false;

		int keyValue = -1;
		if (clicked)
		{
			//BackProject the point
			int depth = pDepthMap[Ycoor*XN_VGA_X_RES+Xcoor];
			double x3D = ((Xcoor-kinectCamera.ox)*depth*FACTOR);
			double y3D = ((Ycoor-kinectCamera.oy)*depth*FACTOR);
			//Print the 3D point
			cout << "3D point: " << x3D << ", " << y3D << ", " << depth <<"; 2D point: " << Xcoor << ", " << Ycoor << endl;

			clicked = false;
			cvWaitKey(0);
		}
		else
		// Control
 		keyValue = cvWaitKey(1);
		if (keyValue==27) bContinueLoop = false;
		if (keyValue==13) trackingMode = BUILD_MODEL;
		if (keyValue==32) flag = true;

		cout << "\r";
	}

	// HOUSEKEEP -------------------------------------------------------------------------	
	context.StopGeneratingAll();
	context.Shutdown();
	cvDestroyAllWindows();
}

double TukeyBiweight(double e, double c)
{
	if(fabs(e)<c)
		return(powf(1.0-powf(e/3.0,2.0),2.0)); //??
	else
		return(0.0);
}

int Plane::frontoPlaneFit(const XnDepthPixel *Depths,Camera *intrinsics, Rect window, float minDepth, float maxDepth)
{
	int NumberOfPoints=0,iMinNew=XN_VGA_Y_RES,jMinNew=XN_VGA_X_RES,iMaxNew=0,jMaxNew=0;
	int iMin=window.y,jMin=window.x,iMax=iMin+window.height,jMax=jMin+window.width;
	double x,y,z,w,e,en,wx,wy,sigmaX,sigmaY;
	double sumXX=0.0,sumXY=0.0,sumXZ=0.0,sumYY=0.0,sumYZ=0.0,sumX=0.0,sumY=0.0,sumZ=0.0,sumW=0.0;
	double FACTOR = intrinsics->pSize/intrinsics->FocalLength;
	
	// Local origin
	int j0=(int)(0.5*(jMax+jMin)), i0=(int)(0.5*(iMax+iMin));
	for(int j=j0;j<jMax;j++) if(Depths[i0*XN_VGA_X_RES+j]!=0) { j0=j; break;}
	double z0 = (double) Depths[i0*XN_VGA_X_RES+j0];
	double x0 = (j0-intrinsics->ox)*z0*FACTOR;
	double y0 = (i0-intrinsics->oy)*z0*FACTOR;
	Matx31d localOrigin(x0,y0,z0);

	// INITIAL FIT
	for (int i=iMin; i<iMax; i++)
	for (int j=jMin; j<jMax; j++)
	{
		// Recover 3D point
		int depth = Depths[i*XN_VGA_X_RES+j];
		w = 1;
		if((depth!=0)&&(depth>minDepth)&&(depth<maxDepth))
		{
			x = ((j-intrinsics->ox)*depth*FACTOR)-x0;
			y = ((i-intrinsics->oy)*depth*FACTOR)-y0;
			z = (double)depth-z0;

			wx = w*x;
			wy = w*y;
			
			sumXX += wx*x;
			sumXY += wx*y;
			sumYY += wy*y;
			sumXZ += wx*z;
			sumYZ += wy*z;
			sumX  += wx;
			sumY  += wy;
			sumZ  += w*z;
			sumW  += w;

			++NumberOfPoints;
		}
	}

	if(sumW==0.0) return -1;

	Matx33d LeftMatrix(sumXX,sumXY,sumX,sumXY,sumYY,sumY,sumX,sumY,sumW);
	Matx31d RightVector(sumXZ,sumYZ,sumZ);
	Matx33d Inverse = LeftMatrix.inv();
	parameters = Inverse * RightVector;

	centroid.val[0]=sumX/sumW;
	centroid.val[1]=sumY/sumW;
	centroid.val[2]=sumZ/sumW;
	centroid += Matx31d(x0,y0,z0);

	residualVariance = 30*30; // 3cm variance - function of z0 in the future? 

	// Resize window
	sigmaX = powf(sumXX/sumW-powf(sumX/sumW,2.0),0.5);
	sigmaY = powf(sumYY/sumW-powf(sumY/sumW,2.0),0.5);
	Point corner1 = intrinsics->imageProject(centroid-Matx31d(sigmaX,sigmaY,0.0)*2.0);
	Point corner2 = intrinsics->imageProject(centroid+Matx31d(sigmaX,sigmaY,0.0)*2.0);
	iMin=corner1.y; jMin=corner1.x; iMax=corner2.y; jMax=corner2.x;
	iMin=max(0,iMin); iMax=min(iMax,XN_VGA_Y_RES);
	jMin=max(0,jMin); jMax=min(jMax,XN_VGA_X_RES);

	// ITERATIVE FIT
	int numIteration=0,MaxIterations=3;

	while (numIteration++<MaxIterations)
	{
		NumberOfPoints = 0;
		sumXX=0.0,sumXY=0.0,sumXZ=0.0,sumYY=0.0,sumYZ=0.0,sumX=0.0,sumY=0.0,sumZ=0.0,sumW=0.0;
		
		for (int i=iMin; i<iMax; i++)
		for (int j=jMin; j<jMax; j++)
		{
			// Recover 3D point
			int depth = Depths[i*XN_VGA_X_RES+j];

			if(depth!=0)
			{
				x = ((j-intrinsics->ox)*depth*FACTOR)-x0;
				y = ((i-intrinsics->oy)*depth*FACTOR)-y0;
				z = (double)depth-z0;

				e = z-(parameters.val[0]*x+parameters.val[1]*y+parameters.val[2]);
				en = e/sqrt(residualVariance);
				w = TukeyBiweight(en,3.0);

				if(w>0.0)
				{
//					if(i<iMinNew) iMinNew=i;
//					if(i>iMaxNew) iMaxNew=i;
//					if(j<jMinNew) jMinNew=j;
//					if(j>jMaxNew) jMaxNew=j;

					wx = w*x;
					wy = w*y;
			
					sumXX += wx*x;
					sumXY += wx*y;
					sumYY += wy*y;
					sumXZ += wx*z;
					sumYZ += wy*z;
					sumX  += wx;
					sumY  += wy;
					sumZ  += w*z;
					sumW  += w;
						
					++NumberOfPoints;
				}
			}
		}

		if(sumW==0.0) return -1;

		LeftMatrix = Matx33d(sumXX,sumXY,sumX,sumXY,sumYY,sumY,sumX,sumY,sumW);
		RightVector = Matx31d(sumXZ,sumYZ,sumZ);
		Inverse = LeftMatrix.inv();
		parameters = Inverse * RightVector;

		centroid.val[0]=sumX/sumW;
		centroid.val[1]=sumY/sumW;
		centroid.val[2]=sumZ/sumW;
		centroid += Matx31d(x0,y0,z0);

		// Resize window
		sigmaX = powf(sumXX/sumW-powf(sumX/sumW,2.0),0.5);
		sigmaY = powf(sumYY/sumW-powf(sumY/sumW,2.0),0.5);
		corner1 = intrinsics->imageProject(centroid-Matx31d(sigmaX,sigmaY,0.0)*2.0);
		corner2 = intrinsics->imageProject(centroid+Matx31d(sigmaX,sigmaY,0.0)*2.0);
		iMin=corner1.y; jMin=corner1.x; iMax=corner2.y; jMax=corner2.x;
		iMin=max(0,iMin); iMax=min(iMax,XN_VGA_Y_RES);
		jMin=max(0,jMin); jMax=min(jMax,XN_VGA_X_RES);

		fitWindow.y=iMin; fitWindow.height = iMax-iMin;
		fitWindow.x=jMin; fitWindow.width  = jMax-jMin;

	}

	// move origin back to kinect coordinate system
	parameters.val[2] -= (parameters.val[0]*x0+parameters.val[1]*y0-z0);

	double norm=sqrt(parameters.val[0]*parameters.val[0]+parameters.val[1]*parameters.val[1]+1.0);
	normal.val[0] = parameters.val[0]/norm;
	normal.val[1] = parameters.val[1]/norm;
	normal.val[2] = -1.0/norm;

	return 0;
}

int Plane::updatePlaneFit(const XnDepthPixel *Depths, const XnRGB24Pixel* rgbMap ,Camera *intrinsics, Rect window, Mat& weighFrame_I, Mat& weighFrame_II, bool flag)
{
	int NumberOfPoints=0,iMin=window.y,jMin=window.x,iMax=iMin+window.height,jMax=jMin+window.width;
	double x,y,z,w,e,en,we,wx,wy, wz,sigmaX,sigmaY, sigmaZ;
	double sumXX=0.0,sumXY=0.0,sumXZ=0.0,sumYY=0.0,sumYZ=0.0,sumX=0.0,sumY=0.0,sumZ=0.0,sumW=0.0,sumEE=0.0, sumZZ = 0.0;
	int varXX, varYY;
	double FACTOR = intrinsics->pSize/intrinsics->FocalLength;

	// Local origin
	float x0 = centroid.val[0];
	float y0 = centroid.val[1];
	float z0 = centroid.val[2];

	//move the plane to new origin
	parameters.val[2] += (parameters.val[0]*x0+parameters.val[1]*y0-z0);

	// ITERATIVE FIT
	int numIteration=0,MaxIterations=3;

		float maxX, minX, maxY, minY;
		maxX = minX = maxY = minY = 0.0;
		bool init = false;

		double stdDev = 0.0;
	while (numIteration++<MaxIterations)
	{
		double sumColorDistWeight = 0.0, sumWeight=0.0;

		NumberOfPoints = 0;
		sumXX=0.0,sumXY=0.0,sumXZ=0.0,sumYY=0.0,sumYZ=0.0,sumX=0.0,sumY=0.0,sumZ=0.0,sumW=0.0, sumZZ = 0.0;
		varXX = varYY = 0.0;

		Point centr2D = intrinsics->imageProject(centroid);
		XnRGB24Pixel centrColor = rgbMap[centr2D.y*XN_VGA_X_RES+centr2D.x];

//		outDebug << "jMin: " << jMin << "jMax: " << jMax << endl;

		for (int i=iMin; i<iMax; i++)
		{
			float* weithPtr_I = weighFrame_I.ptr<float>(i);
			float* weithPtr_II = weighFrame_II.ptr<float>(i);
		for (int j=jMin; j<jMax; j++)
		{
			// Recover 3D point
			int depth = Depths[i*XN_VGA_X_RES+j];
			if(depth!=0)
			{
				XnRGB24Pixel pointColor = rgbMap[i*XN_VGA_X_RES+j];
				float colorDist = powf(centrColor.nBlue-pointColor.nBlue,2) + powf(centrColor.nGreen-pointColor.nGreen,2) + powf(centrColor.nRed-pointColor.nRed,2);
				double wColor = 1;
				if (numIteration > 1)
				{	
					double colorDistN = colorDist/(3*stdDev); 
					wColor = TukeyBiweight(colorDistN, 1.0);
					weithPtr_II[j] = wColor;
				//	cout << "ColorDist: " << colorDist << ", std: " << stdDev << ", Wcolor: " << wColor << endl;
				}
				if (wColor > 0.0)
				{
					x = ((j-intrinsics->ox)*depth*FACTOR)-x0;
					y = ((i-intrinsics->oy)*depth*FACTOR)-y0;
					z = (double)depth-z0;

					e = z-(parameters.val[0]*x+parameters.val[1]*y+parameters.val[2]);
					en = e/sqrt(residualVariance);
					w = TukeyBiweight(en,3.0);
								
					if(w>0.0)
					{
						sumColorDistWeight += (wColor*colorDist);
						sumWeight += wColor;
						
						weithPtr_I[j] = w;
						wx = w*x;
						wy = w*y;
						wz = w*z;
						we = w*e;


						sumXX += wx*x;
						sumXY += wx*y;
						sumYY += wy*y;
						sumXZ += wx*z;
						sumYZ += wy*z;
						sumZZ += wz*z;
						sumEE += we*e;
						sumX  += wx;
						sumY  += wy;
						sumZ  += w*z;
						sumW  += w;

						++NumberOfPoints;
					}
					else weithPtr_I[j]=0.0;
				}
				else weithPtr_I[j]=0.0;
			}
			else weithPtr_I[j]=0.0;
		}
		}
		if(sumW==0.0) return -1;
		Matx33d LeftMatrix = Matx33d(sumXX,sumXY,sumX,sumXY,sumYY,sumY,sumX,sumY,sumW);
		Matx31d RightVector = Matx31d(sumXZ,sumYZ,sumZ);
		Matx33d Inverse = LeftMatrix.inv();
		parameters = Inverse * RightVector;

//		residualVariance = sumEE/sumW; // Proper way to calculate residual variance but template around plane too small when plane is moving fast
		residualVariance = 30*30; // Wrong way! 3cm variance - function of z0 in the future? 

		centroid.val[0]=sumX/sumW;
		centroid.val[1]=sumY/sumW;
		centroid.val[2]=sumZ/sumW;
		
		// Resize window
		sigmaX = powf(sumXX/sumW-powf(sumX/sumW,2.0),0.5);
		sigmaY = powf(sumYY/sumW-powf(sumY/sumW,2.0),0.5);
		sigmaZ = powf(sumZZ/sumW-powf(sumZ/sumW,2.0),0.5); // std in z

		normal.val[0] = parameters.val[0];
		normal.val[1] = parameters.val[1];
		normal.val[2] = -1.0;

		Matx31d yOrt(0,1,0);
		Matx31d xOrt(1,0,0);		

		centroid += Matx31d(x0,y0,z0);

		Matx31d orthogVecXZ = Mat(normal).cross(Mat(yOrt));
		Matx31d orthogVecYZ = Mat(normal).cross(Mat(xOrt));

		//normalize vectors
		double normXZ=sqrt(powf(orthogVecXZ(0),2)+ powf(orthogVecXZ(1),2) + powf(orthogVecXZ(2),2));
		orthogVecXZ(0) = orthogVecXZ(0)/normXZ;
		orthogVecXZ(1) = orthogVecXZ(1)/normXZ;
		orthogVecXZ(2) = orthogVecXZ(2)/normXZ;

		double normYZ=sqrt(powf(orthogVecYZ(0),2)+ powf(orthogVecYZ(1),2) + powf(orthogVecYZ(2),2));
		orthogVecYZ(0) /= normYZ;
		orthogVecYZ(1) /= normYZ;
		orthogVecYZ(2) /= normYZ;

		double multFactX = sigmaX+(abs(sigmaZ*orthogVecXZ(2)));
		double multFactY = sigmaY+(abs(sigmaZ*orthogVecYZ(2)));

		//increase direction vector
		orthogVecXZ(0) *= multFactX;
		orthogVecXZ(1) *= multFactX;
		orthogVecXZ(2) *= multFactX;
		orthogVecYZ(0) *= multFactY;
		orthogVecYZ(1) *= multFactY;
		orthogVecYZ(2) *= multFactY;

		Matx31f rY = centroid+Matx31d(0, orthogVecYZ(1), orthogVecYZ(2))*2;
		Matx31f tY = centroid-Matx31d(0, orthogVecYZ(1), orthogVecYZ(2))*2;

		Matx31f r = centroid+Matx31d(orthogVecXZ(0), 0, orthogVecXZ(2))*2;
		Matx31f t = centroid-Matx31d(orthogVecXZ(0), 0, orthogVecXZ(2))*2;

		corner11 = intrinsics->imageProject(r);
		corner22 = intrinsics->imageProject(t);

		corner11Y = intrinsics->imageProject(rY);
		corner22Y = intrinsics->imageProject(tY);

		Point corner1, corner2;
		corner2.x = corner11.x; corner1.y = corner11Y.y;
		corner1.x = corner22.x; corner2.y = corner22Y.y;

		if (flag) flag = false;
		iMin=corner1.y; jMin=corner1.x; iMax=corner2.y; jMax=corner2.x;
		iMin=max(0,iMin); iMax=min(iMax,XN_VGA_Y_RES);
		jMin=max(0,jMin); jMax=min(jMax,XN_VGA_X_RES);
		
		fitWindow.y=iMin; fitWindow.height = iMax-iMin;
		fitWindow.x=jMin; fitWindow.width  = jMax-jMin;

		stdDev = (sumColorDistWeight/sumWeight)/1.5;
	}
	// move origin back to kinect coordinate system
	parameters.val[2] -= (parameters.val[0]*x0+parameters.val[1]*y0-z0);

	double norm=sqrt(parameters.val[0]*parameters.val[0]+parameters.val[1]*parameters.val[1]+1.0);
	normal.val[0] = parameters.val[0]/norm;
	normal.val[1] = parameters.val[1]/norm;
	normal.val[2] = -1.0/norm;

	return 0;
}



