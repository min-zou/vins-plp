/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sys/time.h>
#include "parameters.h"
#include "feature_manager.h"
#include "line_feature_manager.h"

#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../factor/integration_odom.h"
#include "../factor/odom_factor.h"
#include "../factor/line_local_parameterization.h"
#include "../factor/line_projection_factor.h"
// debug

#define KLTtimeUSE 1

class Estimator
{
public:
    Estimator();
    ~Estimator();
    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputODOM(double t, const Vector3d &linearVelocity, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processODOM(double dt, const Vector3d &linear_velocity, const Vector3d &angular_velocity);
    void processImage_IGNORE(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                      const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &lines, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();
    bool initialStructure(bool use_imu = false, bool use_odom = false);
    bool visualInitialAlign(bool use_imu = false, bool use_odom = false);
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void optimizationwithLine();
    void onlyLineOpt();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                        vector<pair<double, Eigen::Vector3d>> &gyrVector);
    bool getODOMInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &velVector,
                         vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                             Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                             double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    void fastPredictODOM(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    bool ODOMAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
    void pub_odom(const nav_msgs::OdometryConstPtr &first_pose, const nav_msgs::OdometryConstPtr &msg);
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    enum OdomWaitFlag
    {
        Not_Enough = 0,
        Enough = 1
    };
    std::mutex mProcess;
    std::mutex mBuf;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;

    queue<pair<double, Eigen::Vector3d>> velOdomBuf;
    queue<pair<double, Eigen::Vector3d>> gyrOdomBuf;

    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;

    queue<pair<double,
          pair<map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > >,
          map<int, vector<pair<int, Eigen::Matrix<double, 4, 1> > > > > > > point_lineBuf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    OdomWaitFlag odom_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Matrix3d roc;
    Vector3d toc;

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    // odom
    vector<double> odom_dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> odom_linear_vel_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> odom_angular_vel_buf[(WINDOW_SIZE + 1)];
    IntegrationOdom *pre_integra_odom[(WINDOW_SIZE + 1)];
    Vector3d vel_odom_0, gyr_odom_0;
    IntegrationOdom *odom_tmp_preinte;


    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    LineFeatureManager line_f_manager;

    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool first_odom;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_LineFeature[NUM_OF_F][SIZE_LINE];

    double para_Ex_Pose[2][SIZE_POSE];// 左右目到IMU的外参
    double para_Ex_ODOM_Pose[1][SIZE_POSE];// 左目到ODOM的外参
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];
    double scale;
    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;// 时间戳 + frame
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    // odom
    double odom_latest_time;
    Eigen::Vector3d odom_latest_P, odom_latest_V, odom_latest_Ba, odom_latest_Bg, odom_latest_vel_0, odom_latest_gyr_0, result_odom_delta_p, result_odom_delta_v;
    Eigen::Quaterniond odom_latest_Q = Eigen::Quaterniond::Identity(), result_odom_delta_q = Eigen::Quaterniond::Identity();
    Quaterniond odom_delta_q = Eigen::Quaterniond::Identity();
    Vector3d odom_delta_p, odom_delta_v;


    bool initFirstPoseFlag;
    bool initThreadFlag;

private:
    bool updateRefODOM = false;
    double fastOdomPose[WINDOW_SIZE + 1][7];
    bool forceOdomKeyFrame = false;

    cv::Mat img_line;
    std::vector<KeyLine> cur_lines;
};
