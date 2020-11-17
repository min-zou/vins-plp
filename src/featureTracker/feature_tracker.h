/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>
using namespace std;
using namespace camodocal;
using namespace Eigen;
using namespace cv::line_descriptor;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b) {
        return ( a.response > b.response );
    }
};

class FeatureTracker
{
public:
    FeatureTracker();
    void trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 ,
                    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame,
                    map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &linefeatureFrame);

    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file, std::vector<string> un_calib = std::vector<string>());
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    std::vector<std::pair<cv::Point2f, cv::Point2f>> undistortedLinePts(std::vector<std::pair<cv::Point2f, cv::Point2f>> &lines, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    bool lineinSquare(const cv::Point2f &pt);
    void lsd_ex( cv::Mat image, cv::Mat &mask, std::vector<KeyLine>& _keylines);
    void drawKeylines( cv::Mat& inoutImage, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& keylines,
                       const cv::Scalar& color );
    void lineTrack(const cv::Mat &prev_img, const cv::Mat &cur_img,  std::vector<cv::Vec4f> &prev_lines,
                   std::vector<cv::Vec4f> &cur_lines, std::vector<int> &status);
    float ZNCC(cv::Mat lImg, cv::Mat rImg, cv::Point2i lPoint, cv::Point2i rPoint, int size);
    void drawKeylines( cv::Mat& inoutImage, const std::vector<cv::Vec4f>& keylines,
                       const cv::Scalar& color );
    void lineTrackwithZncc(const cv::Mat &prev_img, const cv::Mat &cur_img, const std::vector<cv::Vec4f> &prev_lines,
                           std::vector<cv::Vec4f> &cur_lines, std::vector<int> &match, std::vector<int> &no_match);
    int row, col;
    cv::Mat imTrack;
    cv::Mat mask, line_mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, cur_img_line, prev_img_line;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> cl_pts, pl_pts, cur_un_lines;
    std::vector<cv::Vec4f> nl_pts, cur_lines, prev_lines;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    std::vector<KeyLine> n_kls, prev_kls, cur_kls, cur_un_kls, prev_un_kls;
    vector<int> ids, ids_right, ids_kl;
    vector<int> track_cnt, kl_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    vector<camodocal::CameraPtr> m_un_camera;

    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id, n_kl_id;
    bool hasPrediction;
    cv::Ptr<cv::ximgproc::FastLineDetector> fld;
    cv::Ptr<cv::line_descriptor::LSDDetectorC> lsd;
    cv::line_descriptor::LSDDetectorC::LSDOptions opts;

};
