/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
std::vector<Eigen::Matrix4d> TOC;
Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
std::string ODOM_TOPIC = "/odom";
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int USE_ODOM;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES, UN_CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
int PARALLAX_THRESHOLD = 30;
int USE_FISHEYE_REMAP;
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// 0 无INS 1 IMU 2 ODOM 3 IMU+ODOM
int readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL) {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return -1;
    }
    fclose(fh);
    // std::cerr << R"(     ___           ___           ___           ___                       ___           ___       ___           ___     )" << std::endl;
    // std::cerr << R"(    /\  \         /\  \         /\  \         /\__\          ___        /\  \         /\__\     /\  \         /\__\    )" << std::endl;
    // std::cerr << R"(   /::\  \       /::\  \       /::\  \       /:/  /         /\  \      /::\  \       /:/  /    /::\  \       /::|  |   )" << std::endl;
    // std::cerr << R"(  /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/  /          \:\  \    /:/\ \  \     /:/  /    /:/\:\  \     /:|:|  |   )" << std::endl;
    // std::cerr << R"( /::\~\:\  \   /:/  \:\  \   /:/  \:\  \   /:/__/  ___      /::\__\  _\:\~\ \  \   /:/  /    /::\~\:\  \   /:/|:|__|__ )" << std::endl;
    // std::cerr << R"(/:/\:\ \:\__\ /:/__/ \:\__\ /:/__/ \:\__\  |:|  | /\__\  __/:/\/__/ /\ \:\ \ \__\ /:/__/    /:/\:\ \:\__\ /:/ |::::\__\)" << std::endl;
    // std::cerr << R"(\:\~\:\ \/__/ \:\  \  \/__/ \:\  \ /:/  /  |:|  |/:/  / /\/:/  /    \:\ \:\ \/__/ \:\  \    \/__\:\/:/  / \/__/~~/:/  /)" << std::endl;
    // std::cerr << R"( \:\ \:\__\    \:\  \        \:\  /:/  /   |:|__/:/  /  \::/__/      \:\ \:\__\    \:\  \        \::/  /        /:/  / )" << std::endl;
    // std::cerr << R"(  \:\ \/__/     \:\  \        \:\/:/  /     \::::/__/    \:\__\       \:\/:/  /     \:\  \       /:/  /        /:/  /  )" << std::endl;
    // std::cerr << R"(   \:\__\        \:\__\        \::/  /       ~~~~         \/__/        \::/  /       \:\__\     /:/  /        /:/  /   )" << std::endl;
    // std::cerr << R"(    \/__/         \/__/         \/__/                                   \/__/         \/__/     \/__/         \/__/    )" << std::endl;
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    USE_FISHEYE_REMAP = fsSettings["fisheye_remap"];
    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    USE_ODOM = fsSettings["odom"];
    printf("USE_ODOM: %d\n", USE_ODOM);
    if (USE_ODOM)
    {
        fsSettings["odom_topic"] >> ODOM_TOPIC;
        printf("ODOM_TOPIC: %s\n", ODOM_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];// == 0
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        cv::Mat cv_Tol, cv_Tlc;
        // fsSettings["odom_T_lidar"] >> cv_Tol;
        // fsSettings["lidar_T_cam"] >> cv_Tlc;
        Eigen::Matrix4d T, Tol, Tlc;
        cv::cv2eigen(cv_T, T);

        // cv::cv2eigen(cv_Tol, Tol);
        // cv::cv2eigen(cv_Tlc, Tlc);

        Eigen::Matrix3d r ;
        // Eigen::Vector3d rpy = r.eulerAngles(0, 1, 2);
        // dbg(rpy);
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(-180.0 / 180.0 * 3.1415926, Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ()));

        r = yawAngle * pitchAngle * rollAngle;
        Eigen::Matrix4d Toi = Eigen::Matrix4d::Identity();
        Toi.block(0, 0, 3, 3) = r ;
        Toi.block(0, 3, 3, 1) = Eigen::Vector3d(0.195, 0.0, 0.0);

        // dbg(r);
        // Eigen::Vector3d pt(0, 0, 1);
        // Eigen::Vector3d pt1(1, 0, 0);
        // Eigen::Vector3d pt2(0, 1, 0);

        // dbg(r * pt);
        // dbg(r * pt1);
        // dbg(Tcl.inverse());
        // dbg(r * pt2);
        Eigen::Matrix4d Toc = Toi * T;
        dbg(Toc);
        TOC.push_back(Toc);
        // RIC.push_back(r);

        RIC.push_back(T.block<3, 3>(0, 0));

        // dbg(Toc);
        // RIC.push_back(r);
        // TIC.push_back(Eigen::Vector3d::Zero());
        dbg(T);
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    PARALLAX_THRESHOLD = fsSettings["parallax_threshold"];
    printf("parallax threshold %d\n", PARALLAX_THRESHOLD);

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);
    std::string uncam0Calib;
    fsSettings["un_cam0_calib"] >> uncam0Calib;
    std::string uncam0Path = configPath + "/" + uncam0Calib;
    UN_CAM_NAMES.push_back(uncam0Path);
    if (NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;// imu 到 左目相机的外参
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"]; // 默认 = 0.0
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU && !USE_ODOM)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();

    if (USE_IMU && USE_ODOM)
        return 3;
    else if (USE_IMU)
        return 1;
    else if (USE_ODOM)
        return 2;

    return 0;

}
