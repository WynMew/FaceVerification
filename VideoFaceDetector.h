#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "FaceDetect.h"
#include "FaceFeature.h"
#include "FaceRecog.h"
#include "FaceProcess.h"
#include "config.h"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <string>
#include "dirent.h"
#include <cmath>

#define LEN 5
#define FaceLEN 5

struct FrameRepository {
	cv::Mat item_buffer[LEN];
	cv::Mat IR_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
};

struct FaceRepository {
	cv::Mat item_buffer[LEN];
	cv::Mat IR_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
};

struct VerifyRepository {
	cv::Mat frame_buffer[LEN];
	FR_Rect rect_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
};

struct FaceInTrack
{
	FR_Rect FaceRect;
	cv::Mat matFrame;
	bool bFaceInIR;

	//bool bGotFace; // present track empty?;
};

class VideoFaceDetector
{
public:
	VideoFaceDetector(int &capRGB, int &capIR);
	~VideoFaceDetector();

	FrameRepository gFrameRepository;
	FaceRepository gFaceRepository;
	VerifyRepository gVerifyRepository;
	cv::Mat mFrameNow;

	std::thread VideoCaptureThread();
	std::thread FaceProposerThread();
	std::thread LocalVerifierThread();
	std::thread FaceTrackerThread();
	std::thread FaceSessionCheckerThread();

private:
	void InitCamera(int &capRGB, int &capIR);
	void InitFrameRepository(FrameRepository *ir);
	void InitFaceRepository(FaceRepository *ir);
	void InitFaceSession();
	void GetFrame(); //  Synchronized RGB & IR frame producer
	void ProduceFrameItem(FrameRepository *ir, cv::Mat item, cv::Mat IRimg);
	std::tuple<cv::Mat, cv::Mat> GetFrameItem(FrameRepository *ir);
	double overlap(FR_Rect rectA, FR_Rect rectB);
	void FaceProposer();
	void FaceSessionChecker();
	void ProduceFaceItem(FaceRepository *ir, cv::Mat item);
	void ProduceVerifyItem(VerifyRepository *ir, cv::Mat FrameItem, FR_Rect RectItem);
	std::tuple<cv::Mat, FR_Rect> GetVerifyItem(VerifyRepository *ir);
	cv::Mat GetFaceItem(FaceRepository *ir);
	void CreateUserDB();
	void LocalVerifier();
	void FaceTracker();
	int GetBestFace(FR_Rect Rect[], int &iFaceNum);
	void WriteFaceSession(FR_Rect &Rect, bool bFaceInIR, cv::Mat &imgFrame);
	float calculateSTD(float data[]);

	int iFrameRow;
	int iFrameCol;
	int iFPS;
	int iTrackLostTh;
	cv::VideoCapture vRGB;
	cv::VideoCapture vIR;

	void * FD_handle = NULL;
	void * FF_handle = NULL;
	int isSearch;

	bool bFaceinTrack;
	bool bPoseVarience;
	bool bFaceinIR;
	bool bColorFace;
	float fSearchRatio;

	FaceFeatureDict UserDatabase;
	FaceFeatureDictItem FaceDetectRecogResult;
	FR_Rect EnlargedRect(FR_Rect &InitRect, float &ratio);
	FaceInTrack FaceSession[11]; //  9 faces in one face verification session, last one empty
	//vector is also OK, but may make things too complicated;
	cv::Mat FaceSessionMat[11];
	int iPresentTrackNum;
};
