#ifndef _MORPHOLOGY_H_
#define _MORPHOLOGY_H_
#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Morphology
{
public:
	Morphology(void);
	~Morphology(void);

	void OpenOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H);
	void CloseOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H);
	void GradientOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H);
	void TophatOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H);
	void BlackhatOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H);

private:
	void _DilateOperation(cv::Mat src, cv::Mat dst, int Kernel_W, int Kernel_H);
	void _ErodeOperation(cv::Mat src, cv::Mat dst, int Kernel_W, int Kernel_H);
};

#endif	//_MORPHOLOGY_H_