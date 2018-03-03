#include "Morphology.h"

Morphology::Morphology(void)
{
}

Morphology::~Morphology(void)
{
}

void Morphology::_DilateOperation(cv::Mat src, cv::Mat dst, int Kernel_W, int Kernel_H)
{
	double minv = 0.;
	double maxv = 0.;
	//double* minp = &minv;
	//double* maxp = &maxv;
	cv::Mat srcMat(src.rows, src.cols, CV_32FC1);
	src.convertTo(srcMat, CV_32FC1);
	for (int j=0; j<srcMat.rows; ++j)
	{
		for (int i=0; i<srcMat.cols; ++i)
		{
			if(j<(Kernel_H-1)/2 || j>( srcMat.rows-1-(Kernel_H-1)/2) || i<(Kernel_W-1)/2 || i>( srcMat.cols-1-(Kernel_W-1)/2))
			{
				dst.at<float>(j,i) = srcMat.at<float>(j,i);
			}
			else
			{
				cv::Mat KernelMat(Kernel_H, Kernel_W, CV_32FC1);
				for (int t=-(Kernel_H-1)/2; t<=(Kernel_H-1)/2; ++t)
				{
					for (int k=-(Kernel_W-1)/2; k<=(Kernel_W-1)/2; ++k)
					{				
						KernelMat.at<float>(t+(Kernel_H-1)/2, k+(Kernel_W-1)/2) = srcMat.at<float>(t+j,k+i);
					}
				}
				cv::minMaxIdx(KernelMat, &minv, &maxv);
				dst.at<float>(j,i) = maxv;
				KernelMat.release();
			}

		}
	}
}


void Morphology::_ErodeOperation(cv::Mat src, cv::Mat dst, int Kernel_W, int Kernel_H)
{
	double minv = 0.;
	double maxv = 0.;
	//double* minp = &minv;
	//double* maxp = &maxv;
	cv::Mat srcMat(src.rows, src.cols, CV_32FC1);
	src.convertTo(srcMat, CV_32FC1);
	for (int j=0; j<srcMat.rows; ++j)
	{
		for (int i=0; i<srcMat.cols; ++i)
		{
			if(j<(Kernel_H-1)/2 || j>( srcMat.rows-1-(Kernel_H-1)/2) || i<(Kernel_W-1)/2 || i>( srcMat.cols-1-(Kernel_W-1)/2))
			{
				dst.at<float>(j,i) = srcMat.at<float>(j,i);
			}
			else
			{
				cv::Mat KernelMat(Kernel_H, Kernel_W, CV_32FC1);
				for (int t=-(Kernel_H-1)/2; t<=(Kernel_H-1)/2; ++t)
				{
					for (int k=-(Kernel_W-1)/2; k<=(Kernel_W-1)/2; ++k)
					{				
						KernelMat.at<float>(t+(Kernel_H-1)/2, k+(Kernel_W-1)/2) = srcMat.at<float>(t+j,k+i);
					}
				}
				cv::minMaxIdx(KernelMat, &minv, &maxv);
				dst.at<float>(j,i) = minv;
				KernelMat.release();
			}

		}
	}
}


void Morphology::OpenOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H)
{
	cv::Mat tempMat(dst.rows, dst.cols, dst.type());
	_DilateOperation(src, tempMat, D_Kernel_W, D_Kernel_H);
	_ErodeOperation(tempMat, dst, E_Kernel_W, E_Kernel_H);
}

void Morphology::CloseOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H)
{
	cv::Mat tempMat(dst.rows, dst.cols, dst.type());
	_ErodeOperation(src, tempMat, E_Kernel_W, E_Kernel_H);
	_DilateOperation(tempMat, dst, D_Kernel_W, D_Kernel_H);
}

void Morphology::GradientOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H)
{
	cv::Mat tempMat1(dst.rows, dst.cols, dst.type());
	cv::Mat tempMat2(dst.rows, dst.cols, dst.type());
	_DilateOperation(src, tempMat1, D_Kernel_W, D_Kernel_H);
	_ErodeOperation(src, tempMat2, E_Kernel_W, E_Kernel_H);
	dst = tempMat1-tempMat2;
}

void Morphology::TophatOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H)
{
	cv::Mat srcMat(src.rows, src.cols, CV_32FC1);
	src.convertTo(srcMat, CV_32FC1);
	cv::Mat tempMat(dst.rows, dst.cols, dst.type());
	OpenOperation(src, tempMat, D_Kernel_W, D_Kernel_H, E_Kernel_W, E_Kernel_H);
	dst = srcMat - tempMat;
}

void Morphology::BlackhatOperation(cv::Mat src, cv::Mat dst, int D_Kernel_W, int D_Kernel_H, int E_Kernel_W, int E_Kernel_H)
{
	cv::Mat srcMat(src.rows, src.cols, CV_32FC1);
	src.convertTo(srcMat, CV_32FC1);
	cv::Mat tempMat(dst.rows, dst.cols, dst.type());
	CloseOperation(src, tempMat, D_Kernel_W, D_Kernel_H, E_Kernel_W, E_Kernel_H);
	dst = tempMat - srcMat;
}