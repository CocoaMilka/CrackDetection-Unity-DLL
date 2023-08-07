#pragma once

#ifdef CRACKDETECTION_EXPORTS
#define CRACKDETECTION_API __declspec(dllexport)
#else
#define CRACKDETECTION_API __declspec(dllimport)
#endif


#define NOMINMAX // fucking stupid ass windows.h ruining my code

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "BS_thread_pool.hpp"

/** INTERAL VARIABLES **/
static BS::thread_pool pool(std::thread::hardware_concurrency() - 1);

using namespace cv;


// For passing images between Unity and OpenCV
struct Color32
{
	uchar r;
	uchar g;
	uchar b;
	uchar a;
};


// Functions to export
extern "C"
{
	//__declspec(dllexport) void processCrack(Color32 **raw, int width, int height);
	//__declspec(dllexport) Color32 processCrack();
	__declspec(dllexport) void GetRawImageBytes(unsigned char* data, int width, int height);
	__declspec(dllexport) void processFrame();
}


/** INTERAL FUNCTIONS **/


/// <summary>
/// Creates a line structured element since openCV doesn't include one ?
/// </summary>
/// <param name="length">integer length of the line</param>
/// <param name="angle">integer angle of rotation (must be either 0, 45, 90, 135)!!</param>
/// <returns></returns>
Mat line_strel(int length, int angle);

/// <summary>
/// Filters cracks from an image via morphology operations
/// </summary>
/// <param name="input_image">Mat image to be processed</param>
/// <param name="str_el_size">integer str_el_size (length of structured element for filtering operations)</param>
/// <param name="area_obI">integer area_obI (used for filtering out connected components)</param>
/// <returns>Filtered image highlighting cracks</returns>
Mat crack_detection(Mat input_image, int str_el_size, int area_obI);

/// <summary>
/// Combines proccessed images into a single image, separated to allow for multi threading.
/// This function essentially recursively partiions an image into 4 quadrants and spawns a thread to concurrently compute each quadrant
/// </summary>
/// <param name="T">Mat T1 through T4, images to combine</param>
/// <param name="T1">Mat T1 through T4, images to combine</param>
/// <param name="T2">Mat T1 through T4, images to combine</param>
/// <param name="T3">Mat T1 through T4, images to combine</param>
/// <param name="T4">Mat T1 through T4, images to combine</param>
/// <param name="rLeft">int rLeft to rRight is starting row index to ending row index</param>
/// <param name="rRight">int rLeft to rRight is starting row index to ending row index</param>
/// <param name="cLeft">int cLeft to cRight is starting column index to ending column index</param>
/// <param name="cRight">int cLeft to cRight is starting column index to ending column index</param>
/// <returns></returns>
Mat combineImage(Mat T, Mat T1, Mat T2, Mat T3, Mat T4, int rLeft, int rRight, int cLeft, int cRight);

void combine(Mat T, Mat T1, Mat T2, Mat T3, Mat T4, int rLeft, int rRight, int cLeft, int cRight);