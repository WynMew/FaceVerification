#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "VideoFaceDetector.h"
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;

void main(int argc, char** argv)
{
	int RGBVideoCap = 1;
	int IRVideoCap = 0;

	VideoFaceDetector *VFD = new VideoFaceDetector(RGBVideoCap, IRVideoCap);
	std::thread VideoCaptureTask = VFD->VideoCaptureThread();
	std::thread FaceTrackerTask = VFD->FaceTrackerThread();
	std::thread FaceSessionCheckerTask = VFD->FaceSessionCheckerThread();
	//std::thread FaceProposeTask = VFD->FaceProposerThread();
	std::thread LocalVerifyTask = VFD->LocalVerifierThread();

	VideoCaptureTask.join();
	FaceTrackerTask.join();
	FaceSessionCheckerTask.join();
	//FaceProposeTask.join();
	LocalVerifyTask.join();

	delete VFD;
}
