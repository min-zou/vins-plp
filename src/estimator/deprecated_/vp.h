#pragma once
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core.hpp"
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>

#include <iostream>
#include <map>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <bitset>
#include <time.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Dense>

// #include "precomp_custom.hpp"
using namespace cv;
using namespace line_descriptor;
using namespace Eigen;


struct LineInfo
{
	Vector3d para;
	double length;
	double orientation;
};

class VPDetection
{
public:
	VPDetection(const cv::Point2d &p, double focallength):pp(p),f(focallength)
	{
	};
	~VPDetection(){};

	void run( std::vector<cv::Vec4f> &lines,std::vector<cv::Point3d> &vps, std::vector<std::vector<int> > &clusters ,
		const Eigen::Matrix3d &rcw);
	void getLineParams();

	void getVPHypVia2Lines( std::vector<std::vector<cv::Point3d> >  &vpHypo );

	void getSphereGrids( std::vector<std::vector<double> > &sphereGrid );

	void getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<cv::Point3d> >  &vpHypo, std::vector<cv::Point3d> &vps  );

	void lines2Vps( double thAngle, std::vector<cv::Point3d> &vps, std::vector<std::vector<int> > &clusters );

private:
	// std::vector<std::vector<double> > lines;
	std::vector<cv::Vec4f> lines_;
	std::vector<LineInfo> lineInfos_;
	cv::Point2d pp;
	double f;
	double noiseRatio;

	Eigen::Matrix3d rcw_;
};
