/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

// 参考网址
// https://blog.csdn.net/huanghaihui_123/article/details/103075107
// 重力对齐 ??
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{ -yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    // 通过旋转矩阵R0，使得g=[0, 0, 9.8], !注意，公式中g使用符号，所以这里是正值
    // R0代表了T0时刻imu坐标系0在imu坐标系1的姿态
    return R0;
}
Eigen::Vector4d Utility::pi_from_ppp(Eigen::Vector3d x1, Eigen::Vector3d x2, Eigen::Vector3d x3) {
    Eigen::Vector4d pi;
    pi << ( x1 - x3 ).cross( x2 - x3 ), - x3.dot( x1.cross( x2 ) ); // d = - x3.dot( (x1-x3).cross( x2-x3 ) ) = - x3.dot( x1.cross( x2 ) )

    return pi;
}
void Utility::cvtPluckerToOrthonormal(const Eigen::Vector3d &n, const Eigen::Vector3d &d, double *orth)
{
    //get U matrix
    Eigen::Matrix3d U;
    U.col(0) = n.normalized();
    U.col(1) = d.normalized();
    U.col(2) = n.cross(d).normalized();
    Eigen::Quaterniond qu(U);
    qu.normalize();
    orth[0] = qu.x();
    orth[1] = qu.y();
    orth[2] = qu.z();
    orth[3] = qu.w();

    //get W matrix
    double phi = atan2(d.norm(), n.norm());
    orth[4] = phi;
}

void Utility::cvtPluckerToOrthonormal(const Eigen::Vector3d &n, const Eigen::Vector3d &d, std::vector<double>& orth)
{
    //get U matrix
    Eigen::Matrix3d U;
    U.col(0) = n.normalized();
    U.col(1) = d.normalized();
    U.col(2) = n.cross(d).normalized();
    Eigen::Quaterniond qu(U);
    qu.normalize();
    orth[0] = qu.x();
    orth[1] = qu.y();
    orth[2] = qu.z();
    orth[3] = qu.w();

    //get W matrix
    double phi = atan2(d.norm(), n.norm());
    orth[4] = phi;
}
Eigen::Vector6d Utility::pipi_plk( Eigen::Vector4d pi1, Eigen::Vector4d pi2)
{
    Eigen::Vector6d plk;
    Eigen::Matrix4d dp = pi1 * pi2.transpose() - pi2 * pi1.transpose();
    plk << dp(0, 3), dp(1, 3), dp(2, 3), - dp(1, 2), dp(0, 2), - dp(0, 1);
    return plk;
}
void Utility::cvtOrthonormalToPlucker(double*orth, Eigen::Vector3d &n, Eigen::Vector3d &d)
{

    Eigen::Quaterniond qu(orth[3], orth[0], orth[1], orth[2]);
    double phi = orth[4];

    Eigen::Matrix3d U = qu.toRotationMatrix();
    n = U.col(0) * cos(phi);
    d = U.col(1) * sin(phi);
}

Eigen::Vector6d Utility::plk_from_pose( Eigen::Vector6d plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw )
{

    Eigen::Matrix3d Rwc = Rcw.transpose();
    Eigen::Vector3d twc = -Rwc * tcw;
    return plk_to_pose( plk_c, Rwc, twc);
}
Eigen::Vector6d Utility::plk_to_pose( Eigen::Vector6d plk_w, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw )
{
    Eigen::Vector3d nw = plk_w.head(3);
    Eigen::Vector3d vw = plk_w.tail(3);

    Eigen::Vector3d nc = Rcw * nw + skewSymmetric(tcw) * Rcw * vw;

    Eigen::Vector3d vc = Rcw * vw;
    Eigen::Vector6d plk_c;
    plk_c.head(3) = nc;
    plk_c.tail(3) = vc;
    return plk_c;
}

void Utility::cvtOrthonormalToPlucker(const std::vector<double>& orth, Eigen::Vector3d &n, Eigen::Vector3d &d)
{

    Eigen::Quaterniond qu(orth[3], orth[0], orth[1], orth[2]);
    double phi = orth[4];

    Eigen::Matrix3d U = qu.toRotationMatrix();
    n = U.col(0) * cos(phi);
    d = U.col(1) * sin(phi);
}