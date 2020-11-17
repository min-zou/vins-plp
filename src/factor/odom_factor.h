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
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>

using namespace Eigen;
using namespace std;
template <typename T> inline
void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};

struct RelativeOdomError
{
	RelativeOdomError(double t_x, double t_y, double t_z,
	                  double q_w, double q_x, double q_y, double q_z,
	                  double t_var, double q_var, IntegrationOdom* odom_pre)
		: t_x(t_x), t_y(t_y), t_z(t_z),
		  q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		  t_var(t_var), q_var(q_var), odom_pre(odom_pre) {}
	RelativeOdomError() {}

	template <typename T>
	bool operator()(const T* const w_P_i, const T* const w_P_j, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = w_P_j[0] - w_P_i[0];
		t_w_ij[1] = w_P_j[1] - w_P_i[1];
		t_w_ij[2] = w_P_j[2] - w_P_i[2];

		T w_q_i[4];
		T w_q_j[4];
		w_q_i[0] = w_P_i[6];
		w_q_i[1] = w_P_i[3];
		w_q_i[2] = w_P_i[4];
		w_q_i[3] = w_P_i[5];

		w_q_j[0] = w_P_j[6];
		w_q_j[1] = w_P_j[3];
		w_q_j[2] = w_P_j[4];
		w_q_j[3] = w_P_j[5];

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4], error_ang[3];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

		// residuals[3] = error_q[1] / T(q_var);
		// residuals[4] = error_q[2] / T(q_var);
		// residuals[5] = error_q[3] / T(q_var);

		ceres::QuaternionToAngleAxis(error_q, error_ang);
		residuals[3] = T(2) * error_ang[0] / T(q_var);
		residuals[4] = T(2) * error_ang[1] / T(q_var);
		residuals[5] = T(2) * error_ang[2] / T(q_var);
		Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);

		Eigen::Matrix<T, 6, 6> sqrt_info;
		Eigen::Matrix<double, 6, 6> sqrt_info_double = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(odom_pre->covariance.inverse()).matrixL().transpose();
		for (int i = 0; i < 6; ++i)
		{
			for (int j = 0; j < 6; ++j)
			{
				sqrt_info(i, j) = T(sqrt_info_double(i, j));
			}
		}
		residual = sqrt_info * residual;

		return true;
	}
	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
	                                   const double q_w, const double q_x, const double q_y, const double q_z,
	                                   const double t_var, const double q_var, IntegrationOdom* odom_pre)
	{
		return (new ceres::AutoDiffCostFunction <
		        RelativeOdomError, 6, 7, 7 > (
		            new RelativeOdomError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var, odom_pre)));
	}

	double t_x, t_y, t_z, t_norm;
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;
	IntegrationOdom* odom_pre;

};


struct PlaneError
{

	PlaneError(double t_var, double q_var): t_var(t_var), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const w_P_i, const T* const w_P_j, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = w_P_j[0] - w_P_i[0];
		t_w_ij[1] = w_P_j[1] - w_P_i[1];
		t_w_ij[2] = w_P_j[2] - w_P_i[2];

		T w_q_i[4], ang_i[3];
		T w_q_j[4], ang_j[3];
		w_q_i[0] = w_P_i[6];
		w_q_i[1] = w_P_i[3];
		w_q_i[2] = w_P_i[4];
		w_q_i[3] = w_P_i[5];

		w_q_j[0] = w_P_j[6];
		w_q_j[1] = w_P_j[3];
		w_q_j[2] = w_P_j[4];
		w_q_j[3] = w_P_j[5];
		ceres::QuaternionToAngleAxis(w_q_i, ang_i);
		ceres::QuaternionToAngleAxis(w_q_j, ang_j);

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		residuals[0] = t_i_ij[2] * T(t_var);
		residuals[1] = w_P_i[2] * T(t_var);
		residuals[2] = w_P_j[2] * T(t_var);

		T q_i_j[4], ang_i_j[3];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);
		ceres::QuaternionToAngleAxis(q_i_j, ang_i_j);


		T theta2 = ceres::DotProduct(ang_i_j, ang_i_j);
		T theta = ceres::sqrt(theta2);
		T wx = ang_i_j[0] / theta;
		T wy = ang_i_j[1] / theta;
		T wz = ang_i_j[2] / theta;
		T error[3] ;
		error[0] = wx;
		error[1] = wy;
		error[2] = wz - T(1);

		residuals[3] = T(q_var) * ceres::DotProduct(error, error);

		theta2 = ceres::DotProduct(ang_i, ang_i);
		theta = ceres::sqrt(theta2);
		wx = ang_i[0] / theta;
		wy = ang_i[1] / theta;
		wz = ang_i[2] / theta;
		error[0] = wx;
		error[1] = wy;
		error[2] = wz - T(1);
		residuals[4] = T(q_var) * ceres::DotProduct(error, error);

		theta2 = ceres::DotProduct(ang_j, ang_j);
		theta = ceres::sqrt(theta2);
		wx = ang_j[0] / theta;
		wy = ang_j[1] / theta;
		wz = ang_j[2] / theta;
		error[0] = wx;
		error[1] = wy;
		error[2] = wz - T(1);
		residuals[5] = T(q_var) * ceres::DotProduct(error, error);

		return true;
	}
	static ceres::CostFunction* Create(const double t_var, const double q_var)
	{
		return (new ceres::AutoDiffCostFunction <
		        PlaneError, 6, 7, 7 > (
		            new PlaneError(t_var, q_var)));
	}
	double t_var, q_var;


};


class ODOMFactor : public ceres::SizedCostFunction<6, 7, 7>
{
public:
	ODOMFactor() = delete;
	ODOMFactor(IntegrationOdom* _pre_integration): pre_integration(_pre_integration)
	{
	}
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
	{
		Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
		Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

		Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
		Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
		Eigen::Vector3d delta_p = pre_integration->delta_p;
		Eigen::Quaterniond delta_q = pre_integration->delta_q;

		Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
		residual.block(0, 0, 3, 1) = Qj.matrix().transpose() * (Qi.matrix() * delta_p + Pi - Pj);
		residual.block(3, 0, 3, 1) = 2 * (delta_q.inverse() * (Qi.inverse() * Qj)).vec();
		Eigen::Matrix<double, 6, 6> sqrt_info =
		    Eigen::LLT<Eigen::Matrix<double, 6, 6>>(pre_integration->covariance.inverse()).matrixL().transpose();
		residual = sqrt_info * residual;
		if (jacobians)
		{
			if (jacobians[0])
			{
				Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
				jacobian_pose_i.setZero();
				jacobian_pose_i.block(0, 0, 3, 3) = Qj.matrix().transpose();

				jacobian_pose_i.block(0, 3, 3, 3) = Qj.matrix().transpose() * Qi.matrix() * Utility::skewSymmetric(delta_p);
				jacobian_pose_i.block(3, 3, 3, 3) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(delta_q)).bottomRightCorner<3, 3>();

			}
			if (jacobians[1])
			{
				Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
				jacobian_pose_j.setZero();
				jacobian_pose_j.block(0, 0, 3, 3) = -Qj.matrix().transpose();

				jacobian_pose_j.block(0, 3, 3, 3) = -Utility::skewSymmetric(Qj.matrix().transpose() * Qi.matrix() * delta_p);
				jacobian_pose_j.block(3, 3, 3, 3) = Utility::Qleft(delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

			}
		}
		//
		//
		//

		return true;

	}


	IntegrationOdom* pre_integration;

};

