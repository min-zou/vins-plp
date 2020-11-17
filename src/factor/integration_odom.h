/*******************************************************
 * ODOM预积分
 *
 * add. 2020.07.02
 *
 *
 *
 *******************************************************/


#pragma once

#include "../utility/utility.h"
#include "../estimator/parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationOdom
{
public:
    IntegrationOdom() = delete;
    IntegrationOdom(const Eigen::Vector3d &_vel_0, const Eigen::Vector3d &_gyr_0, const Eigen::Vector3d &_linearized_bg)
        : vel_0{_vel_0}, gyr_0{_gyr_0}, linearized_vel{_vel_0}, linearized_gyr{_gyr_0}, linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 6, 6>::Identity()}, covariance{Eigen::Matrix<double, 6, 6>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        noise = Eigen::Matrix<double, 12, 12>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &vel, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        vel_buf.push_back(vel);
        gyr_buf.push_back(gyr);
        // dbg(dt);
        propagate(dt, vel, gyr);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        vel_0 = linearized_vel;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        // linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], vel_buf[i], gyr_buf[i]);
    }

    void midPointIntegration(double _dt, const Eigen::Vector3d &_vel_0, const Eigen::Vector3d &_gyr_0, const Eigen::Vector3d &_vel_1, const Eigen::Vector3d &_gyr_1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v, bool update_jacobian)
    {

        Vector3d un_vel_0 = delta_q * _vel_0;
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1);
        // result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        result_delta_q = delta_q * Utility::deltaQ(un_gyr * _dt);
        Vector3d un_vel_1 = result_delta_q * _vel_1;
        Vector3d un_vel = 0.5 * (un_vel_0 + un_vel_1);
        result_delta_p = delta_p + un_vel * _dt;
        // result_delta_v = delta_v + un_vel_1-un_vel_0;
        if (update_jacobian)
        {
            Matrix3d R_w_x;

            // 从3维向量转换成反对称矩阵
            R_w_x << 0, -un_gyr(2), un_gyr(1),
                  un_gyr(2), 0, -un_gyr(0),
                  -un_gyr(1), un_gyr(0), 0;


            MatrixXd F = MatrixXd::Zero(6, 6);
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(6, 12);
            V.block<3, 3>(0, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(0, 6) = 0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(3, 3) = 0.5 * Matrix3d::Identity() * _dt;
            V.block<3, 3>(3, 9) = 0.5 * Matrix3d::Identity() * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;// 雅可比更新
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();// 协方差更新
            Eigen::Matrix<double, 6, 6> info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse()).matrixL().transpose();
            if (info(0, 0) > 100000)
            {
                dbg("\r", covariance);

                dbg("\r", info);
            }
        }

    }

    void propagate(double _dt, const Eigen::Vector3d &_vel_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;

        vel_1 = _vel_1;
        gyr_1 = _gyr_1;

        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;

        // 先不更新jacobian
        midPointIntegration(_dt, vel_0, gyr_0, _vel_1, _gyr_1,
                            delta_p, delta_q, delta_v,
                            result_delta_p, result_delta_q, result_delta_v, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        // delta_v = result_delta_v;

        delta_q.normalize();
        sum_dt += dt;
        vel_0 = vel_1;
        gyr_0 = gyr_1;

    }

    // // 真正IMU残差计算!!!!!!
    // Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
    //                                       const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    // {
    //     Eigen::Matrix<double, 15, 1> residuals;

    //     Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    //     Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    //     Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    //     Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    //     Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    //     Eigen::Vector3d dba = Bai - linearized_ba;
    //     Eigen::Vector3d dbg = Bgi - linearized_bg;

    //     // IMU预积分的结果,消除掉acc bias和gyro bias的影响, 对应IMU model中的\hat{\alpha},\hat{\beta},\hat{\gamma}
    //     Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
    //     Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    //     Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    //     // IMU项residual计算,输入参数是状态的估计值, 上面correct_delta_*是预积分值, 二者求'diff'得到residual.
    //     residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    //     residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    //     residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    //     residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    //     residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    //     return residuals;
    // }

    double dt;
    Eigen::Vector3d vel_0, gyr_0;
    Eigen::Vector3d vel_1, gyr_1;

    const Eigen::Vector3d linearized_vel, linearized_gyr;
    Eigen::Vector3d linearized_bg;

    Eigen::Matrix<double, 6, 6> jacobian, covariance;
    Eigen::Matrix<double, 6, 6> step_jacobian;
    Eigen::Matrix<double, 6, 12> step_V;
    Eigen::Matrix<double, 12, 12> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> vel_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

};
