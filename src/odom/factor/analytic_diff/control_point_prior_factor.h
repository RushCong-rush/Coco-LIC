/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <ceres/ceres.h>
#include <sophus_lib/so3.hpp>
#include <utils/sophus_utils.hpp>

namespace cocolic
{
namespace analytic_derivative
{

using SO3d = Sophus::SO3<double>;

class ControlPointPriorFactor : public ceres::SizedCostFunction<6, 4, 3>
{
public:
  ControlPointPriorFactor(const SO3d &rotation_mean,
                          const Eigen::Vector3d &position_mean,
                          const Eigen::Matrix3d &rotation_sqrt_info,
                          const Eigen::Matrix3d &position_sqrt_info)
      : rotation_mean_(rotation_mean),
        position_mean_(position_mean),
        rotation_sqrt_info_(rotation_sqrt_info),
        position_sqrt_info_(position_sqrt_info)
  {
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override
  {
    Eigen::Map<const SO3d> rotation(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> position(parameters[1]);
    const Eigen::Vector3d rotation_error =
        (rotation_mean_.inverse() * rotation).log();

    Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
    residual.head<3>() = rotation_sqrt_info_ * rotation_error;
    residual.tail<3>() =
        position_sqrt_info_ * (position - position_mean_);

    if (!jacobians)
      return true;

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jacobian(
          jacobians[0]);
      jacobian.setZero();
      Eigen::Matrix3d right_jacobian_inverse;
      Sophus::rightJacobianInvSO3(rotation_error, right_jacobian_inverse);
      jacobian.block<3, 3>(0, 0) =
          rotation_sqrt_info_ * right_jacobian_inverse;
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian(
          jacobians[1]);
      jacobian.setZero();
      jacobian.block<3, 3>(3, 0) = position_sqrt_info_;
    }
    return true;
  }

private:
  SO3d rotation_mean_;
  Eigen::Vector3d position_mean_;
  Eigen::Matrix3d rotation_sqrt_info_;
  Eigen::Matrix3d position_sqrt_info_;
};

} // namespace analytic_derivative
} // namespace cocolic
