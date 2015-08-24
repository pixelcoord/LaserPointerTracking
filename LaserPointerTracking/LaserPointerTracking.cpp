#include <string>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp> 

using namespace std;
using namespace cv;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_video249d.lib")
#pragma comment(lib, "opencv_contrib249d.lib")
#pragma comment(lib, "opencv_features2d249d.lib")
#pragma comment(lib, "opencv_nonfree249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_objdetect249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_video249.lib")
#pragma comment(lib, "opencv_contrib249.lib")
#pragma comment(lib, "opencv_features2d249.lib")
#pragma comment(lib, "opencv_nonfree249.lib")
#endif

#define INPUT_FILENAME "laser.mp4"
#define OUTPUT_FILENAME "out.avi"

int MIN_LASER_CIRCLE = 1; //5
int MAX_LASER_CIRCLE = 6; //8
int THRETHOLD_LASER = 255; //240

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

char* window_name = "laser tracking";
const string windowName0 = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string trackbarWindowName = "Trackbars";

vector<cv::Point> points_data;



void on_trackbar(int, void*)
{

}
void createTrackbars()
{
	namedWindow(trackbarWindowName, 0);
	createTrackbar("�ּ� ������ ��Ŭ �ݰ�", trackbarWindowName, &MIN_LASER_CIRCLE, MAX_LASER_CIRCLE, on_trackbar);
	createTrackbar("�ִ� ������ ��Ŭ �ݰ�", trackbarWindowName, &MAX_LASER_CIRCLE, 50, on_trackbar);
	createTrackbar("THRETHOLD_LASER", trackbarWindowName, &THRETHOLD_LASER, 255, on_trackbar);
}

bool compareFrame(Mat &origFrame, Mat &compFrame)
{
	int r, g, b;
	int _r, _g, _b;
	int rows = origFrame.rows;
	int cols = origFrame.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			// �ٱ� �κ��� �Ϻ� �ȼ��� ���� 0���� ����� ���ܽ�Ų��.
			// ī�޶��� �̼��� ��鸲���� �ٱ��κ��� �ȼ��� ���� ���� ���� ����
			if ((i < 20) || ((rows - 20) < i))
			{
				compFrame.at<cv::Vec3b>(i, j)[0] = 0;
				compFrame.at<cv::Vec3b>(i, j)[1] = 0;
				compFrame.at<cv::Vec3b>(i, j)[2] = 0;
				continue;
			}

			if ((j < 20) || ((cols - 20) < j))
			{
				compFrame.at<cv::Vec3b>(i, j)[0] = 0;
				compFrame.at<cv::Vec3b>(i, j)[1] = 0;
				compFrame.at<cv::Vec3b>(i, j)[2] = 0;
				continue;
			}


			
			//ù �����Ӱ� ���� �������� ���Ѵ�.
			//�ֺ� �ȼ��� ȥ���Ѵ�. 
			//-> ī�޶��� �̼��� ��鸲 ������ ���� �����Ӱ� ù �������� �޶����� ���� �����ϱ� ���ؼ�
			b = origFrame.at<cv::Vec3b>(i, j - 2)[0];
			g = origFrame.at<cv::Vec3b>(i, j - 2)[1];
			r = origFrame.at<cv::Vec3b>(i, j - 2)[2];

			b += origFrame.at<cv::Vec3b>(i, j - 1)[0];
			g += origFrame.at<cv::Vec3b>(i, j - 1)[1];
			r += origFrame.at<cv::Vec3b>(i, j - 1)[2];

			b += origFrame.at<cv::Vec3b>(i, j + 0)[0];
			g += origFrame.at<cv::Vec3b>(i, j + 0)[1];
			r += origFrame.at<cv::Vec3b>(i, j + 0)[2];

			b += origFrame.at<cv::Vec3b>(i, j + 1)[0];
			g += origFrame.at<cv::Vec3b>(i, j + 1)[1];
			r += origFrame.at<cv::Vec3b>(i, j + 1)[2];

			b += origFrame.at<cv::Vec3b>(i, j + 2)[0];
			g += origFrame.at<cv::Vec3b>(i, j + 2)[1];
			r += origFrame.at<cv::Vec3b>(i, j + 2)[2];

			b /= 5;
			g /= 5;
			r /= 5;


			_b = compFrame.at<cv::Vec3b>(i, j - 2)[0];
			_g = compFrame.at<cv::Vec3b>(i, j - 2)[1];
			_r = compFrame.at<cv::Vec3b>(i, j - 2)[2];

			_b += compFrame.at<cv::Vec3b>(i, j - 1)[0];
			_g += compFrame.at<cv::Vec3b>(i, j - 1)[1];
			_r += compFrame.at<cv::Vec3b>(i, j - 1)[2];

			_b += compFrame.at<cv::Vec3b>(i, j + 0)[0];
			_g += compFrame.at<cv::Vec3b>(i, j + 0)[1];
			_r += compFrame.at<cv::Vec3b>(i, j + 0)[2];

			_b += compFrame.at<cv::Vec3b>(i, j + 1)[0];
			_g += compFrame.at<cv::Vec3b>(i, j + 1)[1];
			_r += compFrame.at<cv::Vec3b>(i, j + 1)[2];

			_b += compFrame.at<cv::Vec3b>(i, j + 2)[0];
			_g += compFrame.at<cv::Vec3b>(i, j + 2)[1];
			_r += compFrame.at<cv::Vec3b>(i, j + 2)[2];

			_b /= 5;
			_g /= 5;
			_r /= 5;


			//ù �����Ӱ� ���� �������� ���̰� ū ��� �ȼ����� �����ϰ�
			//���̰� ���ٸ� 0���� �����.
			//������ �����Ͱ� ���߰� �ִ� �κ��� ù �����Ӱ� ���� �ٸ� ���̱� ������
			//������� ���� ������ ���� �� �ִ�.
			if (cv::abs<int>(b - _b) > 10 || cv::abs<int>(g - _g) > 10 || cv::abs<int>(r - _r) > 5)
			{
				compFrame.at<cv::Vec3b>(i, j)[0] = compFrame.at<cv::Vec3b>(i, j)[0];
				compFrame.at<cv::Vec3b>(i, j)[1] = compFrame.at<cv::Vec3b>(i, j)[1];
				compFrame.at<cv::Vec3b>(i, j)[2] = compFrame.at<cv::Vec3b>(i, j)[2];
			}
			else
			{
				compFrame.at<cv::Vec3b>(i, j)[0] = 0;
				compFrame.at<cv::Vec3b>(i, j)[1] = 0;
				compFrame.at<cv::Vec3b>(i, j)[2] = 0;
			}
		}
	}

	return false;
}

int main(int argc, char* argv[])
{
	Mat currentFrame;
	Mat threshold;
	Mat HSV;
	bool result = false;

	//Ʈ���� ����
	createTrackbars();


	VideoCapture capture;
	result = capture.open(INPUT_FILENAME);
	if (!result)
	{
		printf("cant open video file. \n");
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	cv::waitKey(1000);


	int frameH = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frameW = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int fps = (int)capture.get(CV_CAP_PROP_FPS);
	int nFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	CvVideoWriter *pWriter = cvCreateVideoWriter(
		OUTPUT_FILENAME,
		CV_FOURCC('D', 'I', 'V', 'X'),
		fps,
		cvSize(frameW, frameH),
		1
		);
	if (NULL == pWriter)
	{
		printf("cant create video writer. \n");
		return -1;
	}




	Mat writeFrame;
	Mat firstFrame;
	capture.read(firstFrame);

	//���� ������ ��ġ ����
	//capture.set(CV_CAP_PROP_POS_FRAMES, capture.get(CV_CAP_PROP_FRAME_COUNT));

	while (1)
	{
		//image to matrix
		capture.read(currentFrame);
		writeFrame = currentFrame;
		if (!writeFrame.data) 
			return -1;
		writeFrame = currentFrame.clone();

		//ù �����Ӱ� ���� ������ ��
		compareFrame(firstFrame, currentFrame);

		//����þ� �� - ������ ������ ��Ŭ�� ã�� ������ �ϱ� ���Ͽ�
		cv::GaussianBlur(currentFrame,
			currentFrame,
			cv::Size(5, 5),
			2.2);

		//BGR to HSV
		vector<Mat> hsv_planes;
		cvtColor(currentFrame, HSV, COLOR_BGR2HSV);
		split(HSV, hsv_planes);
		hsv_planes[0]; // H channel
		hsv_planes[1]; // S channel
		hsv_planes[2]; // V channel

		//THRETHOLD
		unsigned char *input = (unsigned char*)(hsv_planes[2].data);
		int i, j, element;
		int v_threthold = THRETHOLD_LASER;

		for (i = 0; i < hsv_planes[2].rows; i++)
		{
			for (j = 0; j < hsv_planes[2].cols; j++)
			{
				element = input[hsv_planes[2].cols * i + j];

				if (element < v_threthold)
					input[hsv_planes[2].cols * i + j] = 0;
			}
		}

		std::vector<cv::Vec3f> vecCircles;
		std::vector<cv::Vec3f>::iterator itrCircles;

		//����þ� �� - ������ ������ ��Ŭ�� ã�� ������ �ϱ� ���Ͽ�
		cv::GaussianBlur(hsv_planes[2],
			hsv_planes[2],
			cv::Size(5, 5),
			2.2);

		//��Ŭ ã��
		cv::HoughCircles(hsv_planes[2],
			vecCircles,
			CV_HOUGH_GRADIENT,
			1,
			hsv_planes[2].rows / 8,
			1,
			1,
			MIN_LASER_CIRCLE,
			MAX_LASER_CIRCLE);

		//debug code
		//printf("number of circles: %d \n", vecCircles.size());

		
		for (itrCircles = vecCircles.begin(); itrCircles != vecCircles.end(); itrCircles++)
		{
			//������ ������ ��ġ ���� ����
			points_data.push_back(cv::Point((int)(*itrCircles)[0], (int)(*itrCircles)[1]));

			//������ ������ ��Ŭ ����
			cv::circle(writeFrame,
				cv::Point((int)(*itrCircles)[0], (int)(*itrCircles)[1]),
				1,
				cv::Scalar(255, 0, 0),
				CV_FILLED);

			//������ ������ ��Ŭ �ܰ���
			cv::circle(writeFrame,
				cv::Point((int)(*itrCircles)[0], (int)(*itrCircles)[1]),
				(int)(*itrCircles)[2],
				cv::Scalar(64, 255, 64),
				2);
		}

		//������ ������ ������ ��ġ�� ������ �̾ �׸���.
		for (unsigned int i = 1; i < points_data.size(); i++)
		{
			cv::line(writeFrame, points_data[i - 1], points_data[i], cv::Scalar(0, 255, 0, 0));
		}

		//��� �������� ��� ���� ����
		IplImage *pResultImage = new IplImage(writeFrame);
		cvWriteFrame(pWriter, pResultImage);


		//â���� ������ ���
		//imshow(windowName2,threshold);
		imshow(windowName0, writeFrame);
		//imshow(windowName1, currentFrame);
		//imshow(windowName1, HSV);
		//imshow(windowName1, hsv_planes[0]);
		//imshow(windowName2, hsv_planes[2]);
		

		//������ ����ȭ�� ���� ���
		waitKey(30);
	}


	cvReleaseVideoWriter(&pWriter);
	capture.release();


	return 0;
}
