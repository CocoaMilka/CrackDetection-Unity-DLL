#include "pch.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "crackdetection.h"

using namespace cv;
using namespace std;

// Pass by ref to directly modify images
void processCrack(Color32 **raw, int width, int height) 
{
	// Create Mat from passed in struct from Unity
	// Can now modify image via vanilla OpenCV
	Mat frame(height, width, CV_8UC4, *raw);

	// Process image here

	resize(frame, frame, Size(height / 2, width / 2));

	cvtColor(frame, frame, COLOR_BGR2GRAY);

	int str_el_size = 12;
	int area_obI = 110;

	frame = crack_detection(frame, str_el_size, area_obI);

	cout << "Image processed..." << endl;

	// FOR DEBUGGING GRAHH
	//string path = "C:/Users/jrgbk4/Pictures/OPENCV_TESTING/";
	//imwrite("image.png", frame);
}

Mat crack_detection(Mat input_image, int str_el_size, int area_obI)
{
	// Structure elements (lines) these will be used to extract the cracks from the image
	Mat SE1 = line_strel(str_el_size, 0);
	Mat SE2 = line_strel(str_el_size, 45);
	Mat SE3 = line_strel(str_el_size, 90);
	Mat SE4 = line_strel(str_el_size, 135);

	// Temporary filter storing variables
	Mat T1, T2, T3, T4;

	// Filter out cracks using our line structured elements
	// One thread is created to handle each operation, this significantly boosts performance through concurrency

	auto worker1 = pool.submit([&]()
	{
		morphologyEx(input_image, T1, MORPH_OPEN, SE1);
		morphologyEx(T1, T1, MORPH_CLOSE, SE1);
		T1 = max(T1, input_image);
		T1 = T1 - input_image;
	});

	auto worker2 = pool.submit([&]()
	{
		morphologyEx(input_image, T2, MORPH_OPEN, SE2);
		morphologyEx(T2, T2, MORPH_CLOSE, SE2);
		T2 = max(T2, input_image);
		T2 = T2 - input_image;
	});

	auto worker3 = pool.submit([&]()
	{
		morphologyEx(input_image, T3, MORPH_OPEN, SE3);
		morphologyEx(T3, T3, MORPH_CLOSE, SE3);
		T3 = max(T3, input_image);
		T3 = T3 - input_image;
	});

	auto worker4 = pool.submit([&]()
	{
		morphologyEx(input_image, T4, MORPH_OPEN, SE4);
		morphologyEx(T4, T4, MORPH_CLOSE, SE4);
		T4 = max(T4, input_image);
		T4 = T4 - input_image;
	});

	worker1.wait();
	worker2.wait();
	worker3.wait();
	worker4.wait();

	Mat T = Mat::zeros(input_image.rows, input_image.cols, CV_8UC1);
	combineImage(T, T1, T2, T3, T4, 0, input_image.rows, 0, input_image.cols);

	// Convert to binary image and invert, also filters out components that are less than area_obI in size
	threshold(T, T, area_obI, 255, THRESH_BINARY_INV);
	
	return T;
}

Mat combineImage(Mat T, Mat T1, Mat T2, Mat T3, Mat T4, int rLeft, int rRight, int cLeft, int cRight)
{

	if (rLeft < rRight || cLeft < cRight)
	{
		int rMid = (rLeft + rRight) / 2;
		int cMid = (cLeft + cRight) / 2;

		// Each recursively solves a quadrant of the image, divide and conquer technique :3 (Still need to implement recursion)
		auto worker1 = pool.submit([&]()
		{
			combine(T, T1, T2, T3, T4, rLeft, rMid, cLeft, cMid);
		});

		auto worker2 = pool.submit([&]()
		{
			combine(T, T1, T2, T3, T4, rMid, rRight, cLeft, cMid);
		});

		auto worker3 = pool.submit([&]()
		{
			combine(T, T1, T2, T3, T4, rLeft, rMid, cMid, cRight);
		});

		auto worker4 = pool.submit([&]()
		{
			combine(T, T1, T2, T3, T4, rMid, rRight, cMid, cRight);
		});

		worker1.wait();
		worker2.wait();
		worker3.wait();
		worker4.wait();
	}

	return T;
}

void combine(Mat T, Mat T1, Mat T2, Mat T3, Mat T4, int rLeft, int rRight, int cLeft, int cRight)
{
	for (int r = rLeft; r < rRight; r++)
	{
		for (int c = cLeft; c < cRight; c++)
		{
			uchar pixel = max({ T1.at<uchar>(r, c), T2.at<uchar>(r, c), T3.at<uchar>(r, c), T4.at<uchar>(r, c) });
			T.at<uchar>(r, c) = pixel;
		}
	}
}


Mat line_strel(int length, int angle)
{
	// Row vector of 1's with length = # of columns
	Mat line = Mat::ones(1, length, CV_8U);

	switch (angle)
	{
		case 0:
			// return row vector
			break;

		case 45:
			// Creates identity matrix of size = length x length and flips horizontally
			line = Mat::eye(length, length, CV_8U);
			flip(line, line, 1);
			break;

		case 90:
			rotate(line, line, ROTATE_90_CLOCKWISE);
			break;

		case 135:
			// Returns identity matrix of size length x length
			line = Mat::eye(length, length, CV_8U);
			break;

		default:
			// return row vector
			std::cerr << "Please enter either 0, 45, 90, 135. Returning default row vector";
			break;
	}

	return line;
}