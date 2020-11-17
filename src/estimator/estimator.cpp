/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"
// #include "vp.h"
Estimator::Estimator(): f_manager {Rs}, line_f_manager {Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    first_odom = false;
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    if (odom_tmp_preinte != nullptr)
        delete odom_tmp_preinte;

    tmp_pre_integration = nullptr;
    odom_tmp_preinte = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    LineProjectionFactor::sqrt_info = FOCAL_LENGTH / 15 * Matrix2d::Identity();

    td = TD;// time off set 参数 = 0.0
    g = G;// 0.0, 0.0, 9.8
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES, UN_CAM_NAMES); // 加载相机内参

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> linefeatureFrame;

    TicToc featureTrackerTime;

#if KLTtimeUSE == 1
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif

    if (_img1.empty())
        featureTracker.trackImage(t, _img, cv::Mat(), featureFrame, linefeatureFrame); // 单目光流跟踪
    else
        featureTracker.trackImage(t, _img, _img1, featureFrame, linefeatureFrame); //双目处理，光流跟踪
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

#if KLTtimeUSE == 1
    gettimeofday(&end, NULL);
    float time_use_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    std::cout << "KLTtimeUSE = " << time_use_us << " us" << std::endl;
#endif


    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        img_line = featureTracker.cur_img;
        cur_lines = featureTracker.cur_kls;

        pubTrackImage(imgTrack, t);
    }

    if (MULTIPLE_THREAD) // 多线程 默认这里面
    {
        if (inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            point_lineBuf.push(make_pair(t, make_pair(featureFrame, linefeatureFrame))); //把处理后的特征放入buf(queue)中
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        point_lineBuf.push(make_pair(t, make_pair(featureFrame, linefeatureFrame)));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }

}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    fastPredictIMU(t, linearAcceleration, angularVelocity);
    if (solver_flag == NON_LINEAR)
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
}
void Estimator::pub_odom(const nav_msgs::OdometryConstPtr &first_pose, const nav_msgs::OdometryConstPtr &msg)
{
    double t = msg->header.stamp.toSec();
    Eigen::Vector3d init_pose(first_pose->pose.pose.position.x, first_pose->pose.pose.position.y, first_pose->pose.pose.position.z);
    Eigen::Vector3d cur_pose(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond init_q(first_pose->pose.pose.orientation.w, first_pose->pose.pose.orientation.x, first_pose->pose.pose.orientation.y, first_pose->pose.pose.orientation.z);
    Eigen::Quaterniond cur_q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

    Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T1.block(0, 0, 3, 3) = init_q.toRotationMatrix();
    T2.block(0, 0, 3, 3) = cur_q.toRotationMatrix();
    T1.block(0, 3, 3, 1) = init_pose;
    T2.block(0, 3, 3, 1) = cur_pose;
    Eigen::Matrix4d T12 = T1.inverse() * T2;
    Eigen::Vector3d P = T12.block(0, 3, 3, 1);
    Eigen::Matrix3d R = T12.block(0, 0, 3, 3);
    Eigen::Quaterniond Q(R);

    pub_odomtopic_path(P, Q, t);
}
void Estimator::inputODOM(double t, const Vector3d &linearVelocity, const Vector3d &angularVelocity)
{
    mBuf.lock();
    velOdomBuf.push(make_pair(t, linearVelocity));
    gyrOdomBuf.push(make_pair(t, angularVelocity));
    mBuf.unlock();
    // dbg(angularVelocity);

    // 在这里快速计算增量，然后决定
    fastPredictODOM(t, linearVelocity, angularVelocity);
    if (solver_flag == NON_LINEAR) {
        pubLatestOdometry(result_odom_delta_p,
                          result_odom_delta_q,
                          result_odom_delta_v, t);

    }

}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

bool Estimator::getODOMInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &velVector,
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (velOdomBuf.empty())
    {
        dbg("not receive odom\n");
        return false;
    }
    //printf("get odom from %f %f\n", t0, t1);
    //printf("odom fornt time %f   odom end time %f\n", velOdomBuf.front().first, velOdomBuf.back().first);
    if (t1 <= velOdomBuf.back().first)
    {
        while (velOdomBuf.front().first <= t0)
        {
            velOdomBuf.pop();
            gyrOdomBuf.pop();
        }
        while (velOdomBuf.front().first < t1)
        {
            velVector.push_back(velOdomBuf.front());
            velOdomBuf.pop();
            gyrVector.push_back(gyrOdomBuf.front());
            gyrOdomBuf.pop();
        }
        velVector.push_back(velOdomBuf.front());
        gyrVector.push_back(gyrOdomBuf.front());
    }
    else
    {
        printf("wait for odom\n");
        return false;
    }
    return true;
}

// t1 是当前帧处理后的时间加上偏差时间，IMU的最新数据必须大于t1，然后清除小于t0和t1的数据
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if (t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

// 判断条件: 加速度数据非空 且 最新加速度数据大于特征的时间
bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}
bool Estimator::ODOMAvailable(double t)
{
    if (!velOdomBuf.empty() && t <= velOdomBuf.back().first)
    {

        return true;

    }
    else
        return false;
}
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        // pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        pair<double,
             pair<map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > >,
             map<int, vector<pair<int, Eigen::Matrix<double, 4, 1> > > > > >  point_linefeature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        vector<pair<double, Eigen::Vector3d>> odom_velVector, odom_gyrVector;
        if (!point_lineBuf.empty()) //从跟踪中的queue中取出
        {
            point_linefeature = point_lineBuf.front();
            curTime = point_linefeature.first + td;// td = 0
            while (USE_IMU)
            {
                // 如果IMU数据满足要求，则break后向下走
                if ((!USE_IMU  || IMUAvailable(point_linefeature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }

            }
            while (USE_ODOM)
            {
                // 如果IMU数据满足要求，则break后向下走
                if ((!USE_ODOM  || ODOMAvailable(point_linefeature.first + td)))
                    break;
                else
                {
                    // dbg("wait for odom ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(3);
                    std::this_thread::sleep_for(dura);
                }

            }
            mBuf.lock();
            if (USE_IMU)    // prevTime = -1 , curTime = 当前处理特征数据的时间
                getIMUInterval(prevTime, curTime, accVector, gyrVector);//获取区间内的IMU数据

            if (USE_ODOM)
                getODOMInterval(prevTime, curTime, odom_velVector, odom_gyrVector);

            point_lineBuf.pop();//卧操，赋值完就直接扔了
            mBuf.unlock();
            dbg(odom_velVector.size());
            if (USE_IMU)
            {
                if (!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    // 根据加速度的分布，选取不同的积分时间dt
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }

            if (USE_ODOM)
            {
                // odom 不需要重力对齐

                for (size_t i = 0; i < odom_velVector.size(); i++)
                {
                    double dt;
                    if (i == 0)
                        dt = odom_velVector[i].first - prevTime;
                    else if (i == odom_velVector.size() - 1)
                        dt = curTime - odom_velVector[i - 1].first;
                    else
                        dt = odom_velVector[i].first - odom_velVector[i - 1].first;
                    // dbg(dt);
                    processODOM(dt, odom_velVector[i].second, odom_gyrVector[i].second);
                }
                // if (pre_integra_odom[frame_count]->dt_buf.size() <= 2)
                // {
                //     odom_flag = Not_Enough;
                // }
                // else
                // {
                //     odom_flag = Enough;
                // }
            }

            auto time1 = ros::Time::now();

            // cv::Mat out, imgRGB;

            // cv::cvtColor(img_line, imgRGB, cv::COLOR_GRAY2RGB);

            // VPDetection detector;
            // std::vector<cv::Point3d> vps;
            // std::vector<std::vector<int> > clusters ;
            // Eigen::Matrix3d rcw = RIC[0].transpose();
            // detector.run( cur_lines, cv::Point2d(325.656003830114, 235.491374311552), 384.349277679955, vps, clusters , rcw);
            // dbg(cur_lines.back().startPointX);
            // std::vector<KeyLine> temp_ls;
            // int color_id = 0;
            // for (auto kl_ids : clusters)
            // {
            //     temp_ls.clear();
            //     for (auto id : kl_ids)
            //     {
            //         temp_ls.push_back(cur_lines[id]);
            //     }
            //     cv::line_descriptor::drawKeylines(imgRGB, temp_ls, imgRGB);

            //     // if (color_id == 0)
            //     // {
            //     //     cv::line_descriptor::drawKeylines(imgRGB, temp_ls, imgRGB, cv::Scalar( 255, 0,  0));
            //     // }
            //     // else if (color_id == 1)
            //     // {
            //     //     cv::line_descriptor::drawKeylines(imgRGB, temp_ls, imgRGB, cv::Scalar( 0, 255,  0));

            //     // }
            //     // else
            //     // {
            //     //     cv::line_descriptor::drawKeylines(imgRGB, temp_ls, imgRGB, cv::Scalar( 0, 0,  255));

            //     // }
            //     color_id++;
            // }
            // cv::imshow("line-out", imgRGB);
            // cv::waitKey(1);
            auto time2 = ros::Time::now();
            dbg(time2.toSec() - time1.toSec());

            mProcess.lock();
            dbg(point_linefeature.second.second.size());
            processImage(point_linefeature.second.first, point_linefeature.second.second, point_linefeature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(point_linefeature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            pubLinesCloud(*this, header, Eigen::Vector3d(0, 0, 0), Eigen::Matrix3d::Identity());

            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// https://blog.csdn.net/huanghaihui_123/article/details/103075107
// 重力对齐，使得IMU坐标系变成[0, 0, 9.8]
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;// 平均加速度数据
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();// 计算偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{ -yaw, 0, 0}) * R0; // ???
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

// ???
void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

void Estimator::processODOM(double dt, const Vector3d &linear_velocity, const Vector3d &angular_velocity)
{
    if (!first_odom)
    {
        first_odom = true;
        vel_odom_0 = linear_velocity;
        gyr_odom_0 = angular_velocity;
    }

    if (!pre_integra_odom[frame_count])
    {
        // 当滑动窗口为空的话，就添加进入
        pre_integra_odom[frame_count] = new IntegrationOdom{vel_odom_0, gyr_odom_0, Eigen::Vector3d(0, 0, 0)};
    }
    if (frame_count != 0)
    {
        // odom坐标系下
        pre_integra_odom[frame_count]->push_back(dt, linear_velocity, angular_velocity);
        odom_dt_buf[frame_count].push_back(dt);
        odom_linear_vel_buf[frame_count].push_back(linear_velocity);
        odom_angular_vel_buf[frame_count].push_back(angular_velocity);

        odom_tmp_preinte->push_back(dt, linear_velocity, angular_velocity);

        // 世界坐标系下
        int j = frame_count;
        // 计算出来的加速度真值，Bas和Bgs用已经更新过的，并且加上了g分量
        Vector3d un_gyr = 0.5 * (gyr_odom_0 + angular_velocity);
        Vector3d un_vel_0 = Rs[j] * vel_odom_0;// 世界坐标系下的g
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_vel_1 = Rs[j] * linear_velocity;

        // 中值法计算出来的加速度真值
        Ps[j] += dt * 0.5 * (un_vel_0 + un_vel_1) ;
        Vs[j] = un_vel_1;
    }

    vel_odom_0 = linear_velocity;
    gyr_odom_0 = angular_velocity;


    // fastPredictODOM(dt, linear_velocity, angular_velocity);
    // if (solver_flag == NON_LINEAR) {
    //     pubLatestOdometry(result_odom_delta_p,
    //                       result_odom_delta_q,
    //                       result_odom_delta_v, dt);
    // }

}

// https://blog.csdn.net/subiluo/article/details/103697886
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // frame_count 代表当前处理的这一帧在滑动窗口中的第几个, 取值范围是在0到WINDOW_SIZE之间
    if (!pre_integrations[frame_count])
    {
        // 当滑动窗口为空的话，就添加进入
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        //!!!!!!!!!!!!!!!!!!!!! 这个push_back里面进行积分处理
//-----------------------在这个地方计算的只有IMU的部分------------------------//
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

//-----------------------在这个地方计算的对应图像的那个部分------------------------//
        int j = frame_count;
        // 计算出来的加速度真值，Bas和Bgs用已经更新过的，并且加上了g分量
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;// 世界坐标系下的g
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 中值法计算出来的加速度真值
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);// 无运动状态下，上面减去重力分量后，数据接近0
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/*void Estimator::processImage_IGNORE(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 通过视差判断关键帧
    // 当前帧与上一帧比较时差，如果变化很小则删除第二帧，反之则删除最旧的帧。并把这一帧作为新的关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        // 相比于上一帧的odom数量要达到多少帧后再添加
        marginalization_flag = MARGIN_OLD;// 新一帧将被作为关键帧!
        // 删除了最旧的帧，则需要更新参考fast_odom中的参考值
        if (frame_count >= WINDOW_SIZE)
            updateRefODOM = true;

        forceOdomKeyFrame = false;

    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;


    }

    // odom下强制产生关键帧
    if (marginalization_flag && forceOdomKeyFrame)
    {
        forceOdomKeyFrame = false;
        marginalization_flag = MARGIN_OLD;
    }

    ROS_WARN("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());

    // 放入时间戳
    Headers[frame_count] = header;// double数组 大小 10+1

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    imageframe.odom_pre_integration = odom_tmp_preinte;
    all_image_frame.insert(make_pair(header, imageframe));// 加入所有图像序列中  map类型

    dbg(all_image_frame.size());

    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    odom_tmp_preinte = new IntegrationOdom{vel_odom_0, gyr_odom_0, Eigen::Vector3d(0, 0, 0)};

    // !!!!!  我自己屏蔽的
    // 在线外参标定 默认 ESTIMATE_EXTRINSIC = 0
    // if(ESTIMATE_EXTRINSIC == 2)//如果没有外参则进行标定
    // {
    //     ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    //     if (frame_count != 0)
    //     {
    //         //得到当前帧与前一帧之间归一化特征点
    //         vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
    //         Matrix3d calib_ric;
    //         //标定从camera到IMU之间的旋转矩阵
    //         if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
    //         {
    //             ROS_WARN("initial extrinsic rotation calib success");
    //             ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
    //             ric[0] = calib_ric;
    //             RIC[0] = calib_ric;
    //             ESTIMATE_EXTRINSIC = 1;
    //         }
    //     }
    // }

    // 初始化步骤!
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && (USE_IMU || USE_ODOM))
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // ESTIMATE_EXTRINSIC = 0 && 当前帧和初始帧时间相差要大于0.1
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    if (USE_ODOM)
                    {
                        result = initialStructure(false, USE_ODOM);
                    }
                    else
                    {
                        result = initialStructure(USE_IMU);
                    }
                    initial_timestamp = header;
                }
                if (result)
                {
                    solver_flag = NON_LINEAR;
                    onlyLineOpt();
                    // optimization();
                    optimizationwithLine();

                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // monocular + ODOM
        // if (!STEREO && USE_ODOM)
        // {
        //     if (frame_count == WINDOW_SIZE)
        //     {
        //         bool result = false;
        //         // ESTIMATE_EXTRINSIC = 0 && 当前帧和初始帧时间相差要大于0.1
        //         if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
        //         {
        //             result = initialStructure(false, USE_ODOM);
        //             initial_timestamp = header;
        //         }
        //         if (result)
        //         {
        //             solver_flag = NON_LINEAR;
        //             onlyLineOpt();
        //             // optimization();

        //             optimizationwithLine();

        //             slideWindow();
        //             ROS_INFO("Initialization finish!");
        //         }
        //         else
        //             slideWindow();
        //     }
        // }
        // stereo + IMU initilization
        if (STEREO && USE_IMU)
        {
            // 可以认为通过SFM恢复滑动窗中所有帧的位姿
            // 第一次计算时，由于没有任何三维坐标信息，故估计位姿完成不了
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);//求解所有滑动窗的位姿
            // 然后恢复出所有特征点的3D位姿
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);//通过上一步再三角化
            ROS_INFO("~~~~~~~~~~~frame_count = %d", frame_count);
            if (frame_count == WINDOW_SIZE)//图像数量达到滑动窗口的数量时，进行初始化
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);// 陀螺仪bias校正
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    // ???????????????
                    // 为什么ba可以为零呢???
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);// 让IMU从当前时刻开始预积分
                    std::cout << "i = " << i << Bgs[i] << std::endl;
                }
                solver_flag = NON_LINEAR;
                optimization();
                slideWindow();
                ROS_INFO("Initialization finish!");
                //exit(-1);
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else// 正常跟踪
    {
        TicToc t_solve;
        if (!USE_IMU && !USE_ODOM)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        line_f_manager.line_triangulate(Ps, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        onlyLineOpt();
        // optimization();

        optimizationwithLine();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())//跟踪失败
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        line_f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}*/

/**
 * [Estimator::processImage description]
 * @Author   Liu.Xiaox
 * @DateTime 2020-07-23
 * @version  [version]
 * @param    image      [point feature]
 * @param    lines      [line feature]
 * @param    header     [img time]
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                             const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &lines, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    line_f_manager.addFeature(frame_count, lines, td);

    // 通过视差判断关键帧
    // 当前帧与上一帧比较时差，如果变化很小则删除第二帧，反之则删除最旧的帧。并把这一帧作为新的关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        // 相比于上一帧的odom数量要达到多少帧后再添加
        marginalization_flag = MARGIN_OLD;// 新一帧将被作为关键帧!
        // 删除了最旧的帧，则需要更新参考fast_odom中的参考值
        if (frame_count >= WINDOW_SIZE)
            updateRefODOM = true;

        forceOdomKeyFrame = false;

    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
    }

    // odom下强制产生关键帧
    if (marginalization_flag && forceOdomKeyFrame)
    {
        forceOdomKeyFrame = false;
        marginalization_flag = MARGIN_OLD;
    }

    ROS_WARN("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());

    // 放入时间戳
    Headers[frame_count] = header;// double数组 大小 10+1

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    imageframe.odom_pre_integration = odom_tmp_preinte;
    all_image_frame.insert(make_pair(header, imageframe));// 加入所有图像序列中  map类型

    dbg(all_image_frame.size());

    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    odom_tmp_preinte = new IntegrationOdom{vel_odom_0, gyr_odom_0, Eigen::Vector3d(0, 0, 0)};

    // !!!!!  我自己屏蔽的
    // 在线外参标定 默认 ESTIMATE_EXTRINSIC = 0
    // if(ESTIMATE_EXTRINSIC == 2)//如果没有外参则进行标定
    // {
    //     ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    //     if (frame_count != 0)
    //     {
    //         //得到当前帧与前一帧之间归一化特征点
    //         vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
    //         Matrix3d calib_ric;
    //         //标定从camera到IMU之间的旋转矩阵
    //         if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
    //         {
    //             ROS_WARN("initial extrinsic rotation calib success");
    //             ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
    //             ric[0] = calib_ric;
    //             RIC[0] = calib_ric;
    //             ESTIMATE_EXTRINSIC = 1;
    //         }
    //     }
    // }

    // 初始化步骤!
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // ESTIMATE_EXTRINSIC = 0 && 当前帧和初始帧时间相差要大于0.1
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure(USE_IMU);
                    initial_timestamp = header;
                }
                if (result)
                {
                    solver_flag = NON_LINEAR;
                    optimization();
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // monocular + ODOM
        if (!STEREO && (USE_IMU || USE_ODOM))
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // ESTIMATE_EXTRINSIC = 0 && 当前帧和初始帧时间相差要大于0.1
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    if (USE_ODOM)
                    {
                        result = initialStructure(false, USE_ODOM);
                    }
                    else
                    {
                        result = initialStructure(USE_IMU);
                    }
                    initial_timestamp = header;
                }
                if (result)
                {
                    solver_flag = NON_LINEAR;
                    line_f_manager.line_triangulate(Ps, tic, ric);
                    dbg("start op");
                    // onlyLineOpt();
                    dbg(":============:");
                    optimization();
                    // optimizationwithLine();
                    dbg("sadhadashkd");
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if (STEREO && USE_IMU)
        {
            // 可以认为通过SFM恢复滑动窗中所有帧的位姿
            // 第一次计算时，由于没有任何三维坐标信息，故估计位姿完成不了
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);//求解所有滑动窗的位姿
            // 然后恢复出所有特征点的3D位姿
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);//通过上一步再三角化
            ROS_INFO("~~~~~~~~~~~frame_count = %d", frame_count);
            if (frame_count == WINDOW_SIZE)//图像数量达到滑动窗口的数量时，进行初始化
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);// 陀螺仪bias校正
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    // ???????????????
                    // 为什么ba可以为零呢???
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);// 让IMU从当前时刻开始预积分
                    std::cout << "i = " << i << Bgs[i] << std::endl;
                }
                solver_flag = NON_LINEAR;
                optimization();
                slideWindow();
                ROS_INFO("Initialization finish!");
                //exit(-1);
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else// 正常跟踪
    {
        TicToc t_solve;
        if (!USE_IMU && !USE_ODOM)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        line_f_manager.line_triangulate(Ps, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        // optimization();
        onlyLineOpt();
        optimization();
        // optimizationwithLine();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())//跟踪失败
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // line_f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}


// !!!! 只在单目+IMU 初始化中使用 !!!!
bool Estimator::initialStructure(bool use_imu, bool use_odom)
{
    TicToc t_sfm;
    // check imu observibility
    if (use_imu)
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            // a = v / dt
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        // 平均加速度
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 该frame下的加速度与平均加速度的叉乘
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 平均运动加速度小于0.25 m/s^2
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            return false;
        }
    }

    // check odom observibility
    if (use_odom && !use_imu)
    {
        // 要有位移
        map<double, ImageFrame>::iterator frame_it;
        double var_p = 0.0;
        Vector3d move_p;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.odom_pre_integration->sum_dt;
            // a = v / dt
            Vector3d tmp_p = frame_it->second.odom_pre_integration->delta_p;
            Quaterniond tmp_q = frame_it->second.odom_pre_integration->delta_q;
            move_p += tmp_q * tmp_p;
        }

        var_p = sqrt(move_p.transpose() * move_p);

        dbg(var_p);

        if (var_p < 0.035)
        {
            ROS_WARN("ODOM motion not enouth!");
            return false;
        }
    }

    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        // start_frame 对应 frame_count
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        // 如果SFM失败，则将新一帧将被作为关键帧!
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign(use_imu, use_odom))
        return true;
    else
    {
        ROS_INFO("misalign visual structure with INS");

        return false;
    }


}

// 单目下，视觉与IMU联合初始化，获取绝对尺度
bool Estimator::visualInitialAlign(bool use_imu, bool use_odom)
{
    TicToc t_g;
    VectorXd x;
    //solve scale

    bool result = false;
    if (use_imu)
    {
        result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    }
    else if (use_odom)
    {
        result = VisualODOMAlignment(all_image_frame, x);
    }
    else
        return false;

    if (!result)
    {
        ROS_DEBUG("solve s | g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    dbg("scale---", s);
    // for (int i = 0; i <= WINDOW_SIZE; i++)
    // {
    //     pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    // }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    //
    if (use_imu)
    {
        Matrix3d R0 = Utility::g2R(g);
        double yaw = Utility::R2ypr(R0 * Rs[0]).x();
        dbg(yaw);
        // 新的R0代表T0时刻IMU坐标系在理想的IMU坐标系下的姿态
        R0 = Utility::ypr2R(Eigen::Vector3d{ -yaw, 0, 0}) * R0;
        g = R0 * g;
        // Matrix3d rot_diff =  Rs[0].transpose();
        Matrix3d rot_diff = R0;
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];

            Vs[i] = rot_diff * Vs[i];
        }
        ROS_DEBUG_STREAM("g0     " << g.transpose());
        ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());
    }
    if (use_odom)
    {
        Matrix3d rot_diff =  Rs[0].transpose();
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
        }
    }

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    dbg(WINDOW_SIZE);
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 取出0到i之间的匹配点
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);

        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());

            // dbg(average_parallax * 460);

            // 阈值为何?
            if (average_parallax * 460 > PARALLAX_THRESHOLD && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    // 循环滑动窗口中的
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();

        Quaterniond q{Rs[i]};// 转换成四元数
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        // 放置IMU的优化项
        if (USE_IMU)
        {
            // ??
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }

        if (USE_ODOM)
        {

        }

    }

    // 双目 = 2   imu和相机之间 外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    if (USE_ODOM)
    {
        // 只考虑左目与odom的
        para_Ex_ODOM_Pose[0][0] = toc.x();
        para_Ex_ODOM_Pose[0][1] = toc.y();
        para_Ex_ODOM_Pose[0][2] = toc.z();
        Quaterniond q{roc};
        para_Ex_ODOM_Pose[0][3] = q.x();
        para_Ex_ODOM_Pose[0][4] = q.y();
        para_Ex_ODOM_Pose[0][5] = q.z();
        para_Ex_ODOM_Pose[0][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    auto lineorth = line_f_manager.getLineVector();

    for (int i = 0; i < line_f_manager.getFeatureCount(); ++i) {
        para_LineFeature[i][0] = lineorth[i][0];
        para_LineFeature[i][1] = lineorth[i][1];
        para_LineFeature[i][2] = lineorth[i][2];
        para_LineFeature[i][3] = lineorth[i][3];
        para_LineFeature[i][4] = lineorth[i][4];

    }
    // 传感器时差 = 0
    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 =  Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                             para_Pose[0][3],
                                             para_Pose[0][4],
                                             para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] =  rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                         para_Pose[i][1] - para_Pose[0][1],
                                         para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);

        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    //更新线特征
    vector<vector<double>> lineVector = line_f_manager.getLineVector();
    for (int i = 0; i < line_f_manager.getFeatureCount(); i++) {
        for (int j = 0; j < 5; j++)
            lineVector[i][j] = para_LineFeature[i][j];
    }
    line_f_manager.setLineFeature(lineVector);

    if (USE_IMU) // ???
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}
void Estimator::optimizationwithLine()
{
    TicToc t_whole, t_prepare;

    vector2double();//将所有的优化参数转成数据形式，ceres要求
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);//couchy核函数

    //添加ceres参数块
    //因为ceres用的是double数组，所以在下面用vector2double做类型装换
    //Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < WINDOW_SIZE + 1; i++)//添加所有的优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();//这个参数是告诉求解器这个是个单元四元数
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);//7
        if (USE_IMU)
        {
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);//9
        }
    }

    //ESTIMATE_EXTRINSIC!=0则camera到IMU的外参也添加到估计
    for (int i = 0; i < NUM_OF_CAM; i++) {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);//7
        if (!ESTIMATE_EXTRINSIC)//外参如果确定了就不再做优化，如果不确定就需要进一步优化
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    //相机和IMU硬件不同步时估计两者的时间偏差
    if (ESTIMATE_TD) {
        problem.AddParameterBlock(para_Td[0], 1);//居然还有一个时间优化变量 1
        //problem.SetParameterBlockConstant(para_Td[0]);
    }
    // 设置滑窗内第一帧的p和q在优化中不变
    if (true)
        problem.SetParameterBlockConstant(para_Pose[0]);



    dbg(last_marginalization_info);
    //添加边缘化残差
    if (last_marginalization_info && last_marginalization_info->valid) {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        dbg(marginalization_factor);

        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    //添加IMU残差
    if (USE_IMU)
    {
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);//这里面会计算残差以及残差对优化变量雅克比矩阵
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);//这里指定了相关的优化变量
        }
    }
    if (USE_ODOM)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integra_odom[j]->sum_dt > 10.0) // odom 积分间隔限制
                continue;

            Eigen::Vector3d delta_p = pre_integra_odom[j]->delta_p;
            Eigen::Quaterniond delta_q = pre_integra_odom[j]->delta_q;


            ceres::CostFunction * odom_factor = RelativeOdomError::Create(delta_p.x(), delta_p.y(), delta_p.z(),
                                                delta_q.w(), delta_q.x(), delta_q.y(), delta_q.z(),
                                                0.001, 0.01, pre_integra_odom[j]);
            // scale = 1;
            // dbg()
            problem.AddResidualBlock(odom_factor, NULL, para_Pose[i], para_Pose[j]);

            ceres::CostFunction * plane_factor = PlaneError::Create(1.0, 1.0);

            problem.AddResidualBlock(plane_factor, loss_function, para_Pose[i], para_Pose[j]);

        }
    }


    int f_m_cnt = 0;
    int sum_feature = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++sum_feature;
        if (sum_feature > 20)
        {
            break;
        }
    }
    int feature_index = -1;
    if (sum_feature > 20)
    {
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;// 左目的点

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                }

                if (STEREO && it_per_frame.is_stereo)
                {
                    Vector3d pts_j_right = it_per_frame.pointRight;// 右目的点
                    if (imu_i != imu_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    }
                    else
                    {
                        // 左目的点、右目的点、左目点的速度、右目点的速度、？？？
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        // 上述数据、损失函数、左目到IMU外参、右目到IMU外参、???
                        problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    }

                }
                f_m_cnt++;
            }
        }
    }





    //添加线段视觉param block
    int line_feature_index = -1;
    bool debug_flag = false;
    for (auto &it_per_id : line_f_manager.line_feature)//遍历滑窗内所有的空间点
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.line.empty())
        {
            continue;
        }
        ++line_feature_index;
        // for line feature
        ceres::LocalParameterization *line_local_parameterization = new LineLocalParameterization();
        problem.AddParameterBlock(para_LineFeature[line_feature_index], SIZE_LINE, line_local_parameterization); //5
        int imu_j = it_per_id.start_frame - 1;


        for (auto &it_per_frame : it_per_id.line_feature_per_frame) {
            imu_j++;
            Vector3d pts_s = it_per_frame.pts_s;
            Vector3d pts_e = it_per_frame.pts_e;
            // dbg(pts_s);
            // dbg(pts_e);
            LineProjectionFactor *line_f = new LineProjectionFactor(pts_s, pts_e, para_Ex_Pose[0]);
            problem.AddResidualBlock(line_f, loss_function, para_Pose[imu_j], para_LineFeature[line_feature_index]);
            if (debug_flag) {
                ROS_DEBUG("!!!add line residual");
                debug_flag = false;
            }
        }
    }


    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    ceres::Solver::Options options;

    //这一部分是设置
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;

    //这一部分开始求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();//这里会把优化后的结果恢复回去

    TicToc t_whole_marginalization;
    if (frame_count < WINDOW_SIZE)
        return;
    //边缘化处理
    //如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD) {

        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 把数据恢复成原类型
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                        last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                    last_marginalization_parameter_blocks,
                    drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2、将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        {
            if (USE_IMU)
            {
                if (pre_integrations[1]->sum_dt < 10.0) {
                    IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                            vector<double *> {para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}, //优化变量
                            vector<int> {0, 1}); //这里是0,1的原因是0和1是para_Pose[0], para_SpeedBias[0]是需要marg的变量
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
            if (USE_ODOM)
            {
                // if (pre_integra_odom[1]->sum_dt < 10.0) {
                //     IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                //     ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                //             vector<double *> {para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}, //优化变量
                //             vector<int> {0, 1}); //这里是0,1的原因是0和1是para_Pose[0], para_SpeedBias[0]是需要marg的变量
                //     marginalization_info->addResidualBlockInfo(residual_block_info);
                // }
            }

        }

        //3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        if (sum_feature > 20)
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                vector<double *> {para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                vector<int> {0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    /*if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                    vector<double *> {para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                    vector<int> {0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                    vector<double *> {para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                    vector<int> {2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }*/
                }
            }
        }

        //3、将被第零帧观测到的所有普吕克之间，添加到marginalization_info中
        {
            bool debug_flag = true;
            int line_feature_index = -1;
            for (auto &it_per_id : line_f_manager.line_feature) {
                it_per_id.used_num = it_per_id.line_feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))//因为在vector2double里面取para_Line的时候加了这个条件，所以这里必须也要加
                    continue;
                if (!it_per_id.is_triangulation)
                {
                    continue;
                }
                ++line_feature_index;

                if (it_per_id.start_frame == 0) {
                    Vector3d pts_s = it_per_id.line_feature_per_frame[0].pts_s;
                    Vector3d pts_e = it_per_id.line_feature_per_frame[0].pts_e;
                    LineProjectionFactor *line_f = new LineProjectionFactor(pts_s, pts_e, para_Ex_Pose[0]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(line_f, loss_function,
                            vector<double *> {para_Pose[0], para_LineFeature[line_feature_index]},
                            vector<int> {0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }

    //如果次新帧不是关键帧：
    else {
        if (last_marginalization_info &&
                std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                        last_marginalization_parameter_blocks,
                        drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());

}
void  Estimator::onlyLineOpt()
{
    //固定pose， 只优化line的参数，用来调试line的一些参数，看ba优化出来的最好line地图是啥样
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)    // 将窗口内的 p,q 加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // p,q
        // 固定 pose
        problem.SetParameterBlockConstant(para_Pose[i]);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)         // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        // 固定 外参数
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
    vector2double();// 将那些保存在 vector向量里的参数 移到 double指针数组里去

    // 所有特征
    int f_m_cnt = 0;
    int line_feature_index = -1;
    bool debug_flag = true;
    for (auto &it_per_id : line_f_manager.line_feature)//遍历滑窗内所有的空间点
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if (!it_per_id.is_triangulation)
        {
            continue;
        }
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++line_feature_index;
        int imu_j = it_per_id.start_frame - 1;
        ceres::LocalParameterization *line_local_parameterization = new LineLocalParameterization();
        problem.AddParameterBlock(para_LineFeature[line_feature_index], SIZE_LINE, line_local_parameterization); //5


        for (auto &it_per_frame : it_per_id.line_feature_per_frame) {
            imu_j++;
            Vector3d pts_s = it_per_frame.pts_s;
            Vector3d pts_e = it_per_frame.pts_e;
            LineProjectionFactor *line_f = new LineProjectionFactor(pts_s, pts_e, para_Ex_Pose[0]);
            problem.AddResidualBlock(line_f, loss_function, para_Pose[imu_j], para_LineFeature[line_feature_index]);
            if (debug_flag) {
                ROS_DEBUG("!!!add line residual");
                debug_flag = false;
            }
        }
    }

    if (line_feature_index < 3)
    {
        return;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve (options, &problem, & summary);

    //std::cout <<"!!!!!!!!!!!!!onlyLineOpt!!!!!!!!!!!!!\n";
    double2vector();
    //std::cout << summary.FullReport()<<std::endl;

    // line_f_manager.removeLineOutlier(Ps, tic, ric);


}
/*
- 第一部分是那些已经从sliding windows中去掉(marginalize)的节点和特征点构成的约束, 暂且简单的理解为marginalization得到的历史约束的prior, 是一个关于χ的等式约束.
- 第二部分是IMU 运动模型的误差, 每相邻的两帧IMU之间产生一个residual.
- 第三部分是视觉的误差, 单个特征点l在相机cj下的投影会产生一个residual.
*/
// https://blog.csdn.net/huanghaihui_123/article/details/87361621
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    // 数据类型转换，由Ps,Rs转换为para_Pose，Vs,Bas,Bgs转换为para_SpeedBias，tic,ric转换为para_Ex_Pose
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    //添加节点vertex： 7DOF的POSE(SIZE_POSE); 9DOF的IMU(SIZE_SPEEDBIAS)
    //！Step1.1：添加[p,q](7)，[speed,ba,bg](9)
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // SIZE_POSE=7 对应滑动窗口中的相机位置P和旋转Q
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
        {
            // SIZE_SPEEDBIAS=9 对应IMU 速度V加速度计偏置Ba陀螺仪偏置Bg
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
        }

    }
    // 设置滑窗内第一帧的p和q在优化中不变
    if (true)
        problem.SetParameterBlockConstant(para_Pose[0]);


    // 添加节点Vertex 双目 相机参数
    // ? 这里只估计外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {

        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        //
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        //？？ 什么情况下会这样呢??
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) && Vs[frame_count - 1].norm() > 0.3 || openExEstimation)
        {
            ROS_ERROR("estimate extinsic param");
            openExEstimation = 1;// ??
        }
        // 默认进入到下面
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2) // ESTIMATE_TD = 0
        problem.SetParameterBlockConstant(para_Td[0]);// para_Td[0]

    // 边缘化??  没看懂
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    // 添加边（误差项）： IMU
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            // i和j帧的IMU积分、位置、偏置
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }
    if (USE_ODOM)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integra_odom[j]->sum_dt > 10.0) // odom 积分间隔限制
                continue;

            // Eigen::Vector3d delta_p = pre_integra_odom[j]->delta_p;
            // Eigen::Quaterniond delta_q = pre_integra_odom[j]->delta_q;


            // ceres::CostFunction * odom_factor = RelativeOdomError::Create(delta_p.x(), delta_p.y(), delta_p.z(),
            //                                     delta_q.w(), delta_q.x(), delta_q.y(), delta_q.z(),
            //                                     0.5, 0.5, pre_integra_odom[j]);

            ODOMFactor* odom_factor = new ODOMFactor(pre_integra_odom[j]);
            // scale = 1;
            // dbg()
            problem.AddResidualBlock(odom_factor, NULL, para_Pose[i], para_Pose[j]);

            // ceres::CostFunction * plane_factor = PlaneError::Create(70, 50);

            // problem.AddResidualBlock(plane_factor, loss_function, para_Pose[i], para_Pose[j]);

        }
    }
    // 每个特征到每个图像帧的边 重投影误差
    int f_m_cnt = 0;
    // int sum_feature = -1;
    // for (auto &it_per_id : f_manager.feature)
    // {
    //     it_per_id.used_num = it_per_id.feature_per_frame.size();
    //     if (it_per_id.used_num < 4)
    //         continue;

    //     ++sum_feature;
    //     if (sum_feature > 20)
    //     {
    //         break;
    //     }
    // }
    int feature_index = -1;
    // if (sum_feature > 20)
    {
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;// 左目的点

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                }

                if (STEREO && it_per_frame.is_stereo)
                {
                    Vector3d pts_j_right = it_per_frame.pointRight;// 右目的点
                    if (imu_i != imu_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    }
                    else
                    {
                        // 左目的点、右目的点、左目点的速度、右目点的速度、？？？
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        // 上述数据、损失函数、左目到IMU外参、右目到IMU外参、???
                        problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                    }

                }
                f_m_cnt++;
            }
        }
    }

    int line_feature_index = -1;
    bool debug_flag = false;
    for (auto &it_per_id : line_f_manager.line_feature)//遍历滑窗内所有的空间点
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if (!it_per_id.is_triangulation)
        {
            continue;
        }
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++line_feature_index;
        // for line feature
        ceres::LocalParameterization *line_local_parameterization = new LineLocalParameterization();
        problem.AddParameterBlock(para_LineFeature[line_feature_index], SIZE_LINE, line_local_parameterization); //5
        int imu_j = it_per_id.start_frame - 1;

        for (auto &it_per_frame : it_per_id.line_feature_per_frame) {
            imu_j++;
            Vector3d pts_s = it_per_frame.pts_s;
            Vector3d pts_e = it_per_frame.pts_e;
            // dbg(pts_s);
            // dbg(pts_e);
            LineProjectionFactor *line_f = new LineProjectionFactor(pts_s, pts_e, para_Ex_Pose[0]);
            problem.AddResidualBlock(line_f, loss_function, para_Pose[imu_j], para_LineFeature[line_feature_index]);
            if (debug_flag) {
                ROS_DEBUG("!!!add line residual");
                debug_flag = false;
            }
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) // 新一帧将被作为关键帧!
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 把数据恢复成原类型
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                        last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                    last_marginalization_parameter_blocks,
                    drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                        vector<double *> {para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                        vector<int> {0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        if (USE_ODOM)
        {
            if (pre_integra_odom[1]->sum_dt < 10.0) {

                // Eigen::Vector3d delta_p = pre_integra_odom[1]->delta_p;
                // Eigen::Quaterniond delta_q = pre_integra_odom[1]->delta_q;

                // ceres::CostFunction * odom_factor = RelativeOdomError::Create(delta_p.x(), delta_p.y(), delta_p.z(),
                //                                     delta_q.w(), delta_q.x(), delta_q.y(), delta_q.z(),
                //                                     0.001, 0.001, pre_integra_odom[1]);
                ODOMFactor* odom_factor = new ODOMFactor(pre_integra_odom[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(odom_factor, NULL,
                        vector<double *> {para_Pose[0], para_Pose[1]}, //优化变量
                        vector<int> {0}); //这里是0,1的原因是0和1是para_Pose[0], para_SpeedBias[0]是需要marg的变量
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // if(sum_feature>20)
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                vector<double *> {para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                vector<int> {0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                    vector<double *> {para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                    vector<int> {0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                    vector<double *> {para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                    vector<int> {2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }
        {
            bool debug_flag = true;
            int line_feature_index = -1;
            for (auto &it_per_id : line_f_manager.line_feature) {
                it_per_id.used_num = it_per_id.line_feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))//因为在vector2double里面取para_Line的时候加了这个条件，所以这里必须也要加
                    continue;
                if (!it_per_id.is_triangulation)
                {
                    continue;
                }
                ++line_feature_index;

                if (it_per_id.start_frame == 0) {
                    Vector3d pts_s = it_per_id.line_feature_per_frame[0].pts_s;
                    Vector3d pts_e = it_per_id.line_feature_per_frame[0].pts_e;
                    LineProjectionFactor *line_f = new LineProjectionFactor(pts_s, pts_e, para_Ex_Pose[0]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(line_f, loss_function,
                            vector<double *> {para_Pose[0], para_LineFeature[line_feature_index]},
                            vector<int> {0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
                std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                        last_marginalization_parameter_blocks,
                        drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
                if (USE_ODOM)
                {
                    std::swap(pre_integra_odom[i], pre_integra_odom[i + 1]);

                    odom_dt_buf[i].swap(odom_dt_buf[i + 1]);
                    odom_angular_vel_buf[i].swap(odom_angular_vel_buf[i + 1]);
                    odom_linear_vel_buf[i].swap(odom_linear_vel_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);

                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            if (USE_ODOM)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];

                delete pre_integra_odom[WINDOW_SIZE];
                pre_integra_odom[WINDOW_SIZE] = new IntegrationOdom{vel_odom_0, gyr_odom_0, Eigen::Vector3d(0, 0, 0)};

                odom_dt_buf[WINDOW_SIZE].clear();
                odom_angular_vel_buf[WINDOW_SIZE].clear();
                odom_linear_vel_buf[WINDOW_SIZE].clear();
            }
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            if (USE_ODOM)
            {
                for (unsigned int i = 0; i < odom_dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = odom_dt_buf[frame_count][i];
                    Vector3d tmp_linear = odom_linear_vel_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = odom_angular_vel_buf[frame_count][i];

                    pre_integra_odom[frame_count - 1]->push_back(tmp_dt, tmp_linear, tmp_angular_velocity);

                    odom_dt_buf[frame_count - 1].push_back(tmp_dt);
                    odom_angular_vel_buf[frame_count - 1].push_back(tmp_angular_velocity);
                    odom_linear_vel_buf[frame_count - 1].push_back(tmp_linear);
                }

                Vs[frame_count - 1] = Vs[frame_count];

                delete pre_integra_odom[WINDOW_SIZE];
                pre_integra_odom[WINDOW_SIZE] = new IntegrationOdom{vel_odom_0, gyr_odom_0, Eigen::Vector3d(0, 0, 0)};
                odom_dt_buf[WINDOW_SIZE].clear();
                odom_angular_vel_buf[WINDOW_SIZE].clear();
                odom_linear_vel_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
    line_f_manager.removeFront(frame_count);//移除线特征

}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
    line_f_manager.removeBack();//因为线特征没有深度的概念，因此可以直接移除

}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

// 在边缘化中，如果第一帧被边缘化掉了，则需要更新参考位姿
// 如果相比于上一帧，一直没有插入新帧，到时间了强制产生关键帧
void Estimator::fastPredictODOM(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity)
{
    // 1. delta time
    double dt = t - odom_latest_time;
    // dbg(odom_latest_time);
    if (odom_latest_time == 0.0)
    {
        dt = 0;
    }

    odom_latest_time = t;
    Vector3d un_vel_0 = result_odom_delta_q * odom_latest_vel_0;
    Vector3d un_gyr = 0.5 * (odom_latest_gyr_0 + angular_velocity);
    // result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    result_odom_delta_q = result_odom_delta_q * Utility::deltaQ(un_gyr * dt);
    Vector3d un_vel_1 = result_odom_delta_q * linear_velocity;
    Vector3d un_vel = 0.5 * (un_vel_0 + un_vel_1);
    result_odom_delta_p = result_odom_delta_p + un_vel * dt;
    result_odom_delta_v = result_odom_delta_v + un_vel;
    // dbg(result_odom_delta_q.toRotationMatrix());
    // dbg(odom_latest_gyr_0);
    // dbg(linear_velocity);
    odom_latest_vel_0 = linear_velocity;
    odom_latest_gyr_0 = angular_velocity;
    // dbg(result_odom_delta_p);

}

// 使用上一时刻姿态快速预测PVQ姿态
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    // 1. delta time
    double dt = t - latest_time;
    latest_time = t;
    // 2. wRc * (acc - ba) - g // 转换到世界坐标系下，减去重力
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    // 3. 上一角速度和当前角速度的中值减去偏置
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    // 4. 计算当前的Q
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    // 5. 当前线加速度转换到世界坐标系下
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    // 6. 均值一下线加速度
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 7. P = P + V*dt + 0.5*a*dt^2
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    // 8. 当前的速度 V = V + V*dt
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;

    queue<pair<double, Eigen::Vector3d>> tmp_odom_velBuf = velOdomBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_odom_gyrBuf = gyrOdomBuf;

    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        // fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    while (!tmp_odom_velBuf.empty())
    {
        double t = tmp_odom_velBuf.front().first;
        Eigen::Vector3d vel = tmp_odom_velBuf.front().second;
        Eigen::Vector3d gyr = tmp_odom_gyrBuf.front().second;
        // fastPredictODOM(t, vel, gyr);
        tmp_odom_velBuf.pop();
        tmp_odom_gyrBuf.pop();
    }
    mBuf.unlock();
}
