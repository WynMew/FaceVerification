#include "VideoFaceDetector.h"

typedef struct FrameRepository FrameRepository;
typedef struct FaceRepository FaceRepository;
typedef struct FaceInTrack FaceInTrack;
typedef struct VerifyRepository VerifyRepository;

//FrameRepository VideoFaceDetector::gFrameRepository;
//FaceRepository VideoFaceDetector::gFaceRepository;
//
//cv::Mat VideoFaceDetector::mFrameNow;
//cv::VideoCapture VideoFaceDetector::vRGB;
//cv::VideoCapture VideoFaceDetector::vIR;

using namespace cv;
using namespace std;

VideoFaceDetector::VideoFaceDetector(int &capRGB, int &capIR)
{
	InitCamera(capRGB, capIR);
	isSearch = 1; //0注册不需要检索，1非注册状态需要检索用户ID
	iFPS = 60;
	iTrackLostTh = 5;

	bFaceinTrack = false;
	bPoseVarience = false;
	bFaceinIR = false;
	bColorFace = false;
	fSearchRatio = 1.6;
	iPresentTrackNum = 0;

	int flagFD = 0;
	int flagFF = 0;
	flagFD = FD_Init(FD_handle, "CPU", "model/ldmk.bin");
	flagFF = FF_Init(FF_handle, "CPU", "model/fea.bin");

	CreateUserDB();
	UserDatabase.load("userdatabase.dat");
	InitFrameRepository(&gFrameRepository);
	InitFaceRepository(&gFaceRepository);
	//InitFaceSession();
}

void VideoFaceDetector::InitCamera(int &capRGB, int &capIR)
{
	iFrameRow = 480; // default resolution
	iFrameCol = 640;

	vRGB = cv::VideoCapture(capRGB); // 0 for RGB camere
	vRGB.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	vRGB.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	vRGB.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);

	vIR = cv::VideoCapture(capIR); // 1 for IR Camera
	vIR.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	vIR.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	//cap.set(CV_CAP_PROP_EXPOSURE, -3); 
	vIR.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);
	//double exposure = cap.get(CV_CAP_PROP_AUTO_EXPOSURE);
	//cout << exposure << endl;
}

void VideoFaceDetector::InitFrameRepository(FrameRepository *ir)
{
	ir->write_position = 0;
	ir->read_position = 0;
	cv::Mat matrix = cv::Mat(iFrameCol, iFrameRow, CV_32F, cv::Scalar::all(0));// transposed img size
	for (int i = 0; i < LEN; i++)
	{
		ir->item_buffer[i] = matrix.clone(); // write in zeros. actually we don't need this.
		ir->IR_buffer[i] = matrix.clone();
	}
}

void VideoFaceDetector::InitFaceRepository(FaceRepository *ir)
{
	ir->write_position = 0;
	ir->read_position = 0;
}

void VideoFaceDetector::GetFrame()
{
	cv::Mat RGBimg;
	cv::Mat IRimg;
	while (1)
	{//non stop request to get highest FPS from cam hardware.
		vRGB >> RGBimg;
		vIR >> IRimg;
		if (!RGBimg.empty() && !IRimg.empty())
		{
			transpose(RGBimg, RGBimg);
			transpose(IRimg, IRimg);
			/*  debug  /*
			//hconcat(RGBimg, IRimg, mFrameNow);
			//imshow("Frame Now", mFrameNow);
			//cvWaitKey(1);
			*/
			ProduceFrameItem(&gFrameRepository, RGBimg, IRimg);// write into frame repo
		}
	}
}

FR_Rect VideoFaceDetector::EnlargedRect(FR_Rect &InitRect, float &ratio) //ratio >1
{
	if (ratio <= 1)
	{
		return InitRect;
	}
	else
	{
		FR_Rect outputRect;
		outputRect.width = unsigned int(InitRect.width * ratio);
		outputRect.height = unsigned int(InitRect.height * ratio);

		if (InitRect.top > unsigned int(InitRect.height * (ratio - 1) / 2)) // overflow exception
		{
			outputRect.top = InitRect.top - unsigned int(InitRect.height * (ratio - 1) / 2);
		}
		else
		{
			outputRect.top = 1;
		}
		
		if (InitRect.left > unsigned int(InitRect.width * (ratio - 1) / 2))// overflow exception
		{
			outputRect.left = InitRect.left - unsigned int(InitRect.width * (ratio - 1) / 2);
		}
		else
		{
			outputRect.left = 1;
		}


		if ((outputRect.top + outputRect.height) > ( iFrameCol - 1))
		{
			outputRect.height = iFrameCol - outputRect.top - 1;
		}

		if ((outputRect.left + outputRect.width) > (iFrameRow - 1))
		{
			outputRect.width = iFrameRow - outputRect.left - 1;
		}

		//std::cout << outputRect.top << " " << outputRect.left << " " << outputRect.height << " " << outputRect.width << std::endl;
		return outputRect;
	}
}

int VideoFaceDetector::GetBestFace(FR_Rect Rect[], int &iFaceNum)
{
	int iTmpQ = 0;
	int iID = 0;
	for (int i = 0; i < iFaceNum; i++)
	{
		if (Rect[i].nQuality > iTmpQ)
		{
			iTmpQ = Rect[i].nQuality;
			iID = i;
		}
	}
	return iID;
}

void VideoFaceDetector::FaceTracker()
{
	FR_Rect RGBRectSel; // Selected best face in frame
	FR_Rect RGBRectROI; // enlarged area for face search
	int iFaceTrackLostCounter = 0;

	while (1) // nonstop searching or tracking
	{
		//-- Get frame block --//
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 / (iFPS + 20)));// set the disp FPS here,  better waiting for frames insted of waiting for empty slot.
		cv::Mat img, IRimg, OrigFrame;
		tie(img, IRimg) = GetFrameItem(&gFrameRepository);
		OrigFrame = img.clone();
		//-- Tracker --//
		if (!bFaceinTrack) // non face detected in last frame or ROI (global parameter)
		{
			FR_Rect RGBRect[50]; // 50 faces at most. May fail if num of face in frame > 50
			int iFaceNumRGB = FD_Detect(FD_handle, img.data, img.cols, img.rows, RGBRect);
			//std::cout << "face searching in frame\n";

			if (iFaceNumRGB > 0){
				int iID = GetBestFace(RGBRect, iFaceNumRGB);
				RGBRectSel = RGBRect[iID]; // best face in frame
				RGBRectROI = EnlargedRect(RGBRectSel, fSearchRatio);
				rectangle(img, cv::Rect(RGBRectSel.left, RGBRectSel.top, RGBRectSel.width, RGBRectSel.height), 
					Scalar(0, 0, 255, 0), 3, 8, 0);
				//std::cout << "face detected in frame\n";
				bFaceinTrack = true;
				//int iFaceNumRGB = FD_Detect(FD_handle, IRimg.data, IRimg.cols, IRimg.rows, IRRect);

				cv::Mat IRROI = IRimg(Rect(RGBRectROI.left, RGBRectROI.top, RGBRectROI.width, RGBRectROI.height));//Actually a persuade crop
				cv::Mat IRROIc = IRROI.clone();
				FR_Rect IRROIRect[10];// dummy array, just used for function return
				int iFaceNumIRROI = FD_Detect(FD_handle, IRROIc.data, IRROIc.cols, IRROIc.rows, IRROIRect);
				bool bFaceInCorrIR;
				iFaceNumIRROI > 0 ? bFaceInCorrIR = true : bFaceInCorrIR = false;
				if (iPresentTrackNum < 10)
				{
					WriteFaceSession(RGBRectSel, bFaceInCorrIR, OrigFrame);
					//FaceSessionMat[iPresentTrackNum] = ???
				}
				iPresentTrackNum++;
			}

			iFaceTrackLostCounter = 0;
		}
		else // get ROI in frame and re-detect
		{
			FR_Rect ROIRect[20]; // 20 faces at most. May fail if num of face in ROI > 20
			cv::Mat imgROI = img(Rect(RGBRectROI.left, RGBRectROI.top, RGBRectROI.width, RGBRectROI.height));//Actually a persuade crop
			cv::Mat imgROIc = imgROI.clone(); // You must clone it, so that the source data can be moved to dst data. 
			//std::cout << "face search in ROI\n";

			//cv::imshow("face search ROI", imgROIc);
			//cvWaitKey(1);

			int iFaceNumROI = FD_Detect(FD_handle, imgROIc.data, imgROIc.cols, imgROIc.rows, ROIRect);
			//std::cout <<"# of face(s) found in ROI:"<< iFaceNumROI << std::endl;

			if (iFaceNumROI > 0){
				int iID = GetBestFace(ROIRect, iFaceNumROI);
				RGBRectSel = ROIRect[iID]; // best face in frame
				RGBRectSel = ROIRect[iID]; // best face in ROI
				RGBRectSel.left = RGBRectSel.left + RGBRectROI.left;
				RGBRectSel.top = RGBRectSel.top + RGBRectROI.top; //face in ROI to in frame
				RGBRectROI = EnlargedRect(RGBRectSel, fSearchRatio);//update ROI
				rectangle(img, cv::Rect(RGBRectSel.left, RGBRectSel.top, RGBRectSel.width, RGBRectSel.height),
					Scalar(0, 0, 255, 0), 3, 8, 0);
				//std::cout << "face detected in ROI\n";
				iFaceTrackLostCounter = 0;

				cv::Mat IRROI = IRimg(Rect(RGBRectROI.left, RGBRectROI.top, RGBRectROI.width, RGBRectROI.height));//Actually a persuade crop
				cv::Mat IRROIc = IRROI.clone();
				FR_Rect IRROIRect[10];// dummy array, just used for function return
				int iFaceNumIRROI = FD_Detect(FD_handle, IRROIc.data, IRROIc.cols, IRROIc.rows, IRROIRect);
				bool bFaceInCorrIR;
				iFaceNumIRROI > 0 ? bFaceInCorrIR = true : bFaceInCorrIR = false;
				if (iPresentTrackNum < 10)
					WriteFaceSession(RGBRectSel, bFaceInCorrIR, OrigFrame);
				iPresentTrackNum++;
			}
			else if (iFaceTrackLostCounter < iTrackLostTh)
			{
				iFaceTrackLostCounter++;
				//std::cout << "iFaceTrackLostCounter = " << iFaceTrackLostCounter<< std::endl;
			}
			else
			{
				bFaceinTrack = false;
				std::cout << "face lost\n";
			}
		}

		cv::Mat mAlignedCam;
		cv::hconcat(img, IRimg, mAlignedCam);
		cv::imshow("Frame with face detection", mAlignedCam);
		cvWaitKey(1);
	}
}

void VideoFaceDetector::FaceSessionChecker()
{
	while (1){
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 / (iFPS))); // do not check too frequently
		if (iPresentTrackNum == 10) // track cache full
		{
			int iIRFaceCounter = 0;
			float dataYaw[10];
			float dataPitch[10];
			float fSTDYaw;
			float fSTDPitch;
			float fMeanQuality = 0;
			float fMeanClarity = 0;
			//float fMeanBrightness = 0;
			int TrackSize = 10;
			int iTmpQ = 0;
			int iID = 0;

			for (int i = 0; i < TrackSize; i++)
			{
				if (FaceSession[i].bFaceInIR)
					iIRFaceCounter ++;
				dataYaw[i] = FaceSession[i].FaceRect.yaw;
				dataPitch[i] = FaceSession[i].FaceRect.pitch;
				fMeanQuality += FaceSession[i].FaceRect.nQuality;
				fMeanClarity += FaceSession[i].FaceRect.clarity;

				if (FaceSession[i].FaceRect.nQuality > iTmpQ)
				{
					iTmpQ = FaceSession[i].FaceRect.nQuality;
					iID = i;
				}
				//fMeanBrightness = FaceSession[i].brightness;
			}
			
			fMeanQuality = fMeanQuality / TrackSize;
			fMeanClarity = fMeanClarity / TrackSize;
			//fMeanBrightness = fMeanBrightness / TrackSize;

			fSTDYaw = calculateSTD(dataYaw);
			fSTDPitch = calculateSTD(dataPitch);
			float fMaxPose;
			fSTDYaw > fSTDPitch ? fMaxPose = fSTDYaw : fMaxPose = fSTDPitch;

			...

			uint8_t* pixelPtr = (uint8_t*)matFace.data;
			int cn = matFace.channels();
			Scalar_<uint8_t> bgrPixel;

			long lB = 0;
			long lG = 0;
			long lR = 0;
			float fDataColor[3];
			float fColorVarience;

			try
			{
				...
			}
			catch (...)
			{
				
			}

			//std::cout << lB << " " << lG << " " << lR << std::endl;
			//std::cout << fColorVarience << std::endl;
			//std::cout << fMaxPose << "-" << iIRFaceCounter << "-" << fMeanQuality << "-" << fMeanClarity << "-" << fColorVarience << std::endl;

			if (...)
			{
				//std::cout << "living face\n";
				//ExtractAndSearch(FaceSession[iID].FaceRect, FaceSession[iID].matFrame, UserDatabase, FaceDetectRecogResult, FF_handle, isSearch);
				//if (FaceDetectRecogResult.DeepBlueID >= 0)
				//rectangle(FaceSession[iID].matFrame, cv::Rect(FaceSession[iID].FaceRect.left, FaceSession[iID].FaceRect.top, 
				//	FaceSession[iID].FaceRect.width, FaceSession[iID].FaceRect.height),	Scalar(0, 0, 255, 0), 3, 8, 0);
				//cv::imshow("Frame", FaceSession[iID].matFrame);
				//cvWaitKey(1);
				//std::cout << "User ID " << FaceDetectRecogResult.DeepBlueID << " detected " << std::endl;
				ProduceVerifyItem(&gVerifyRepository, FaceSession[iID].matFrame, FaceSession[iID].FaceRect);
			}
			else
			{
				std::cout << "spoofing face ?\n";
			}

			iPresentTrackNum = 0; //re-initialize face track session. 
		}
	}
}


void VideoFaceDetector::FaceProposer()
{
	cv::Mat mFrameP = cv::Mat(iFrameCol, iFrameRow, CV_32F, cv::Scalar::all(0));// transposed img size
	cv::Mat mFramePP = cv::Mat(iFrameCol, iFrameRow, CV_32F, cv::Scalar::all(0));// transposed img size
	FR_Rect rectP;
	FR_Rect rectPP;
	FR_Rect rectNow;
	FR_Rect rectEnl;
	cv::Mat mGrayMatP;
	cv::Mat mGrayMatPP;
	cv::Mat mAlignedCam;
	int facecountP;
	int facecountPP;
	float ratioEnlarge = 1.5;

	int iCounter = 0;

	while (1){
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 / (iFPS+20)));// set the disp FPS here,  better waiting for frames insted of waiting for empty slot.
		//cv::Mat img = GetFrameItem(&gFrameRepository);

		cv::Mat img, IRimg;
		tie(img, IRimg) = GetFrameItem(&gFrameRepository);

		FR_Rect rect[50];
		FR_Rect IRrect[50];
		//std::cout<<"Before " << clock() << std::endl;
		int facecountA = FD_Detect(FD_handle, img.data, img.cols, img.rows, rect);
		//std::cout <<"After" <<clock() << std::endl;  //~ 20 ms per frame

		int iTmpQ = 0;
		int iID = 0;
		for (int i = 0; i < facecountA; i++)
		{
			//std::cout << "FaceID:" << i << " Face Quality: " << rect[i].nQuality << std::endl;
			rectangle(img, cv::Rect(rect[i].left, rect[i].top, rect[i].width, rect[i].height), Scalar(0, 0, 255, 0), 3, 8, 0);

			//std::ostringstream str;
			//str << "ID:" << i << " Quality: " << rect[i].nQuality << std::endl;
			//cv::putText(img, str.str(), cv::Point(rect[i].left, rect[i].top), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 250));

			// search the best face:
			if (rect[i].nQuality > iTmpQ)
			{
				iTmpQ = rect[i].nQuality;
				iID = i;
			}
		}
		rectNow = rect[iID];

		int facecountIR = FD_Detect(FD_handle, IRimg.data, IRimg.cols, IRimg.rows, IRrect);
		int iIRTmpQ = 0;
		int iIRID = 0;
		for (int i = 0; i < facecountIR; i++)
		{
			rectangle(IRimg, cv::Rect(IRrect[i].left, IRrect[i].top, IRrect[i].width, IRrect[i].height), Scalar(0, 0, 255, 0), 3, 8, 0);

			if (IRrect[i].nQuality > iIRTmpQ)
			{
				iIRTmpQ = IRrect[i].nQuality;
				iIRID = i;
			}
		}
		FR_Rect rectIR = IRrect[iIRID];

		//std::cout << facecountIR << " face(s) detected in IR img \n";

		std::ostringstream str; //do defination/initilization in every loop !
		str << rectNow.pitch << " " << rectNow.roll << " " << rectNow.yaw;
		cv::putText(img, str.str(), cv::Point(10, 20), CV_FONT_HERSHEY_PLAIN,1.4, cv::Scalar(255, 0, 255));

		cv::hconcat(img, IRimg, mAlignedCam);
		cv::imshow("Frame with face detection & pose", mAlignedCam);
		cvWaitKey(1);

		rectEnl = EnlargedRect(rectIR, ratioEnlarge);

		auto time = std::chrono::system_clock::now();
		std::time_t nowtime = std::chrono::system_clock::to_time_t(time);

		if (countNonZero(mGrayMatP) > 0 && countNonZero(mGrayMatPP) > 0 && facecountP > 0 && facecountPP >0 && facecountIR >0)
		{
			if (overlap(rectNow, rectP) > 0.5 && overlap(rectP, rectPP) > 0.5)
			{
				if (overlap(rectNow, rectIR) > 0.25)
				{
					cv::Mat mFace = img(Rect(rectNow.left, rectNow.top, rectNow.width, rectNow.height));
					/*imshow("proposed face", mFace);
					cvWaitKey(1);
					*/
					if (iCounter == 0)
					{
						/*
						ProduceFaceItem(&gFaceRepository, mFace);
						ExtractAndSearch(rect[iID], img, UserDatabase, FaceDetectRecogResult, FF_handle, isSearch);
						//if (FaceDetectRecogResult.DeepBlueID >= 0)
						std::cout << "Proposer User ID " << FaceDetectRecogResult.DeepBlueID << " detected " << std::endl;
						*/
						ProduceVerifyItem(&gVerifyRepository, img, rect[iID]);
					}
					iCounter++;
				}
				else
				{
					std::cout << "Spoofing? \n";
				}
			}
			else
			{
				iCounter = 0;
				std::cout << "overlap < th \n";
			}
		}
		else
		{
			iCounter = 0;
			std::cout << std::ctime(&nowtime) << "No living face detected \n";
		}

		mFrameP = img.clone();
		rectP = rectNow;
		facecountP = facecountA;
		mFramePP = mFrameP.clone();
		rectPP = rectP;
		facecountPP = facecountP;

		cv::cvtColor(mFrameP, mGrayMatP, cv::COLOR_BGR2GRAY);
		mGrayMatPP = mGrayMatP.clone();

	}
}

void VideoFaceDetector::ProduceFrameItem(FrameRepository *ir, cv::Mat item, cv::Mat IRimg)
{
	unique_lock<mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % LEN) == ir->read_position)
	{ // item buffer is full, just wait here.
		//std::cout << "FrameProducer is waiting for an empty slot...\n";
		(ir->repo_not_full).wait(lock); // 生产者等待"frame库缓冲区不为满"这一条件发生.
	}

	(ir->item_buffer)[ir->write_position] = item.clone();
	(ir->IR_buffer)[ir->write_position] = IRimg.clone();
	//cout << "write at " << ir->write_position << endl;
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == LEN) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者frame库不为空.
	lock.unlock();
}

void VideoFaceDetector::ProduceFaceItem(FaceRepository *ir, cv::Mat item)
{
	unique_lock<mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % LEN) == ir->read_position)
	{ // item buffer is full, just wait here.
		std::cout << "FaceProducer is waiting for an empty slot...\n";
		(ir->repo_not_full).wait(lock); // 生产者等待"frame库缓冲区不为满"这一条件发生.
	}

	(ir->item_buffer)[ir->write_position] = item.clone();
	//cout << "write at " << ir->write_position << endl;
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == LEN) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者frame库不为空.
	lock.unlock();
}

cv::Mat VideoFaceDetector::GetFaceItem(FaceRepository *ir)
{
	unique_lock<mutex> lock(ir->mtx);
	// item buffer is empty, just wait here.
	//cout << "get face, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
	while (ir->write_position == ir->read_position)
	{
		//std::cout << "FrameConsumer is waiting for frames...\n";
		(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
	}
	cv::Mat data = (ir->item_buffer)[ir->read_position].clone();
	//cout << "read at " << ir->read_position << endl;
	(ir->read_position)++; // 读取位置后移
	if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
		ir->read_position = 0;

	(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
	lock.unlock(); // 解锁.
	return data; // 返回frame.
}

std::tuple<cv::Mat, cv::Mat> VideoFaceDetector::GetFrameItem(FrameRepository *ir)
{
	unique_lock<mutex> lock(ir->mtx);
	// item buffer is empty, just wait here.
	//cout << "get frame, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
	while (ir->write_position == ir->read_position)
	{
		//std::cout << "FrameConsumer is waiting for frames...\n";
		(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
	}
	cv::Mat RGBImg = (ir->item_buffer)[ir->read_position].clone();
	cv::Mat IRImg = (ir->IR_buffer)[ir->read_position].clone();
	//cout << "read at " << ir->read_position << endl;
	(ir->read_position)++; // 读取位置后移
	if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
		ir->read_position = 0;

	(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
	lock.unlock(); // 解锁.
	return  std::make_tuple(RGBImg, IRImg);
}

void VideoFaceDetector::ProduceVerifyItem(VerifyRepository *ir, cv::Mat FrameItem, FR_Rect RectItem)
{
	unique_lock<mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % LEN) == ir->read_position)
	{ // item buffer is full, just wait here.
		std::cout << "FaceProducer is waiting for an empty slot...\n";
		(ir->repo_not_full).wait(lock); // 生产者等待"frame库缓冲区不为满"这一条件发生.
	}

	(ir->frame_buffer)[ir->write_position] = FrameItem.clone();
	(ir->rect_buffer)[ir->write_position] = RectItem;
	//cout << "write at " << ir->write_position << endl;
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == LEN) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者frame库不为空.
	lock.unlock();
}

std::tuple<cv::Mat, FR_Rect> VideoFaceDetector::GetVerifyItem(VerifyRepository *ir)
{
	unique_lock<mutex> lock(ir->mtx);
	// item buffer is empty, just wait here.
	//cout << "get frame, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
	while (ir->write_position == ir->read_position)
	{
		//std::cout << "FrameConsumer is waiting for frames...\n";
		(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
	}
	cv::Mat RGBImg = (ir->frame_buffer)[ir->read_position].clone();
	FR_Rect Rect = (ir->rect_buffer)[ir->read_position];
	//cout << "read at " << ir->read_position << endl;
	(ir->read_position)++; // 读取位置后移
	if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
		ir->read_position = 0;

	(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
	lock.unlock(); // 解锁.
	return  std::make_tuple(RGBImg, Rect);
}

double VideoFaceDetector::overlap(FR_Rect rectA, FR_Rect rectB) // IOU
{
	double x_overlap = max(0, int(min(rectA.left + rectA.width, rectB.left + rectB.width) - max(rectA.left, rectB.left)));
	double y_overlap = max(0, int(min(rectA.top + rectA.height, rectB.top + rectB.height) - max(rectA.top, rectB.top)));
	double overlapArea = x_overlap * y_overlap;
	double sizeA = rectA.width * rectA.height;
	double sizeB = rectB.width * rectB.height;
	double fOverLap = overlapArea / (sizeA + sizeB - overlapArea);
	return fOverLap;
}

float VideoFaceDetector::calculateSTD(float data[])
{
	int iDataSize = sizeof(data) / sizeof(data[0]);
	float sum = 0.0, mean, standardDeviation = 0.0;
	for (int i= 0; i <iDataSize; ++i)
		sum += data[i];
	mean = sum / iDataSize;
	for (int i = 0; i < iDataSize; ++i)
		standardDeviation += pow(data[i] - mean, 2);
	return sqrt(standardDeviation / iDataSize);
}

void VideoFaceDetector::CreateUserDB()
{
	int facetotal = 0;
	FR_Rect rect[50];

	std::cout << "Processing user database\n";
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir("FaceDataBase\\")) != NULL) { // set face data base dir here.
		int i = 0;
		while ((ent = readdir(dir)) != NULL) {
			//do face stuff:
			stringstream ssImgPath;
			string sImgPath;
			ssImgPath << "FaceDataBase\\";
			ssImgPath << ent->d_name;
			ssImgPath >> sImgPath;
			std::cout << sImgPath << std::endl;

			cv::Mat img = imread(sImgPath);
			facetotal = FD_Detect(FD_handle, img.data, img.cols, img.rows, rect);
			if (facetotal > 0)
			{
				//std::cout << "Before " << clock() << std::endl;
				FF_FeaExtract(FF_handle, img.data, img.cols, img.rows, FaceDetectRecogResult.feature, rect[0]);
				//std::cout << "After " << clock() << std::endl; // 50 ~ 100 ms ; 720p 
				FaceDetectRecogResult.DeepBlueID = i;
				UserDatabase.AddPerson(FaceDetectRecogResult);
				printf("%d finish\n", i);
				i++;
			}
			std::cout << "num of db items: " << UserDatabase.GetTotal() << std::endl;
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}
	UserDatabase.save("userdatabase.dat");
}

void VideoFaceDetector::LocalVerifier()
{
	//cv::Mat mFace;
	cv::Mat mFrame;
	FR_Rect Rect;

	while (1)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 / iFPS)); // 100 FPS

		//mFace = GetFaceItem(&gFaceRepository);
		//cv::imshow("verifier face", mFace);
		//cvWaitKey(1);

		// TODO: local verifier to propose feature.

		tie(mFrame, Rect) = GetVerifyItem(&gVerifyRepository);

		FR_Rect LocalRect[50];
		int FaceinlocalFrame = FD_Detect(FD_handle, mFrame.data, mFrame.cols, mFrame.rows, LocalRect); // redetect
		ExtractAndSearch(LocalRect[0], mFrame, UserDatabase, FaceDetectRecogResult, FF_handle, isSearch);
		//ExtractAndSearch(Rect, mFrame, UserDatabase, FaceDetectRecogResult, FF_handle, isSearch);

		//if (FaceDetectRecogResult.DeepBlueID >= 0)
		//rectangle(mFrame, cv::Rect(Rect.left, Rect.top, Rect.width, Rect.height),	Scalar(0, 0, 255, 0), 3, 8, 0);
		std::cout << "verifier User ID " << FaceDetectRecogResult.DeepBlueID << " detected " << std::endl;
		//cv::imshow("Frame", mFrame);
		//cvWaitKey(1);

	}
}

void VideoFaceDetector::InitFaceSession() // don't need this
{
	int iFaceSessionSize = sizeof(FaceSession) / sizeof(FaceSession[0]);
	iPresentTrackNum = 0;

	for (int i = 0; i < iFaceSessionSize; i++)
	{
		//FaceSession[i].bGotFace = false;
		//std::cout << i << " " << FaceSession[i].bGotFace << std::endl;
	}
}

void VideoFaceDetector::WriteFaceSession(FR_Rect &Rect, bool bFaceInIR, cv::Mat &imgFrame)
{
	//FaceSession[iPresentTrackNum].roll = Rect.roll;
	//FaceSession[iPresentTrackNum].yaw = Rect.yaw;
	//FaceSession[iPresentTrackNum].pitch = Rect.pitch;
	//FaceSession[iPresentTrackNum].width = Rect.width;
	//FaceSession[iPresentTrackNum].height = Rect.height;
	//FaceSession[iPresentTrackNum].nQuality = Rect.nQuality;
	//FaceSession[iPresentTrackNum].clarity = Rect.clarity;
	//FaceSession[iPresentTrackNum].brightness = Rect.brightness;
	FaceSession[iPresentTrackNum].FaceRect = Rect;
	FaceSession[iPresentTrackNum].bFaceInIR = bFaceInIR;
	FaceSession[iPresentTrackNum].matFrame = imgFrame.clone();
	//std::cout << FaceSession[iPresentTrackNum].nQuality << std::endl;
	//std::cout << FaceSession[iPresentTrackNum].roll << " " << FaceSession[iPresentTrackNum].yaw << " " << FaceSession[iPresentTrackNum].pitch << " " << FaceSession[iPresentTrackNum].bFaceInIR << std::endl;
}

std::thread VideoFaceDetector::VideoCaptureThread()
{
	return std::thread([this] {this->GetFrame(); });
}

std::thread VideoFaceDetector::FaceProposerThread()
{
	return std::thread([this] {this->FaceProposer(); });
}

std::thread VideoFaceDetector::LocalVerifierThread()
{
	return std::thread([this] {this->LocalVerifier(); });
}

std::thread VideoFaceDetector::FaceTrackerThread()
{
	return std::thread([this] {this->FaceTracker(); });
}

std::thread VideoFaceDetector::FaceSessionCheckerThread()
{
	return std::thread([this] {this->FaceSessionChecker(); });
}
VideoFaceDetector::~VideoFaceDetector()
{
	FD_Destroy(FD_handle);
	FF_Destroy(FF_handle);
}
