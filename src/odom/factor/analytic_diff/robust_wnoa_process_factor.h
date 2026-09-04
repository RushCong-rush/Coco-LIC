/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <ceres/ceres.h>
#include <spline/trajectory.h>
#include <odom/factor/analytic_diff/rd_spline_view.h>
#include <odom/factor/analytic_diff/so3_spline_view.h>

#include <array>
#include <stdexcept>

namespace cocolic {
namespace analytic_derivative {
namespace robust_wnoa {

using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using SO3 = Sophus::SO3d;

constexpr double kCauchyLossScale = 2.449489742783178;

inline ceres::LossFunction* MakeCauchyLoss(double cost_scale = 1.0) {
  if (cost_scale < 0.0)
    throw std::invalid_argument("WNOA robust cost scale must be non-negative");
  ceres::LossFunction* cauchy = new ceres::CauchyLoss(kCauchyLossScale);
  if (cost_scale == 1.0)
    return cauchy;
  return new ceres::ScaledLoss(cauchy, cost_scale, ceres::TAKE_OWNERSHIP);
}

inline Mat6 BuildSqrtInformation(double delta_t, double spectral_density) {
  if (!(delta_t > 0.0) || !(spectral_density > 0.0))
    throw std::invalid_argument("WNOA interval and spectral density must be positive");

  const Mat3 density = spectral_density * Mat3::Identity();
  Mat6 covariance;
  covariance.topLeftCorner<3, 3>() =
      (delta_t * delta_t * delta_t / 3.0) * density;
  covariance.topRightCorner<3, 3>() =
      (delta_t * delta_t / 2.0) * density;
  covariance.bottomLeftCorner<3, 3>() =
      covariance.topRightCorner<3, 3>();
  covariance.bottomRightCorner<3, 3>() = delta_t * density;

  Eigen::LLT<Mat6> decomposition(covariance);
  if (decomposition.info() != Eigen::Success)
    throw std::invalid_argument("WNOA covariance must be positive definite");
  return decomposition.matrixL().solve(Mat6::Identity());
}

inline double SegmentDuration(const Trajectory& trajectory,
                              size_t support_start) {
  return (trajectory.knts.at(support_start + 4)
          - trajectory.knts.at(support_start + 3)) * NS_TO_S;
}

}  // namespace robust_wnoa

class RobustWnoaTranslationFactor : public ceres::CostFunction {
 public:
  RobustWnoaTranslationFactor(const Trajectory& trajectory,
                              size_t support_start,
                              double spectral_density,
                              const robust_wnoa::Mat6& residual_sqrt_weight =
                                  robust_wnoa::Mat6::Identity())
      : RobustWnoaTranslationFactor(
            robust_wnoa::SegmentDuration(trajectory, support_start),
            trajectory.blending_mats.at(support_start), spectral_density,
            residual_sqrt_weight) {}

  RobustWnoaTranslationFactor(double delta_t,
                              const robust_wnoa::Mat4& blending,
                              double spectral_density,
                              const robust_wnoa::Mat6& residual_sqrt_weight =
                                  robust_wnoa::Mat6::Identity())
      : delta_t_(delta_t),
        blending_(blending),
        residual_sqrt_weight_(residual_sqrt_weight),
        sqrt_information_(robust_wnoa::BuildSqrtInformation(
            delta_t, spectral_density)) {
    set_num_residuals(6);
    for (int i = 0; i < 4; ++i)
      mutable_parameter_block_sizes()->push_back(3);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const std::pair<int, double> segment_start(0, 0.0);
    const std::pair<int, double> segment_end(0, 1.0);
    RdSplineView::JacobianStruct position_i_jacobian;
    RdSplineView::JacobianStruct position_j_jacobian;
    RdSplineView::JacobianStruct velocity_i_jacobian;
    RdSplineView::JacobianStruct velocity_j_jacobian;
    const bool compute_jacobians = jacobians != nullptr;

    const robust_wnoa::Vec3 position_i = RdSplineView::evaluateNURBS(
        segment_start, blending_, parameters,
        compute_jacobians ? &position_i_jacobian : nullptr);
    const robust_wnoa::Vec3 position_j = RdSplineView::evaluateNURBS(
        segment_end, blending_, parameters,
        compute_jacobians ? &position_j_jacobian : nullptr);
    const robust_wnoa::Vec3 velocity_i = RdSplineView::velocityNURBS(
        segment_start, delta_t_, blending_, parameters,
        compute_jacobians ? &velocity_i_jacobian : nullptr);
    const robust_wnoa::Vec3 velocity_j = RdSplineView::velocityNURBS(
        segment_end, delta_t_, blending_, parameters,
        compute_jacobians ? &velocity_j_jacobian : nullptr);

    robust_wnoa::Vec6 raw_residual;
    raw_residual.head<3>() =
        position_j - position_i - delta_t_ * velocity_i;
    raw_residual.tail<3>() = velocity_j - velocity_i;
    Eigen::Map<robust_wnoa::Vec6> residual(residuals);
    residual = residual_sqrt_weight_ * sqrt_information_ * raw_residual;

    if (!compute_jacobians) return true;
    for (int i = 0; i < 4; ++i) {
      if (!jacobians[i]) continue;
      Eigen::Matrix<double, 6, 3> raw_jacobian;
      raw_jacobian.topRows<3>() =
          (position_j_jacobian.d_val_d_knot[i]
           - position_i_jacobian.d_val_d_knot[i]
           - delta_t_ * velocity_i_jacobian.d_val_d_knot[i])
          * robust_wnoa::Mat3::Identity();
      raw_jacobian.bottomRows<3>() =
          (velocity_j_jacobian.d_val_d_knot[i]
           - velocity_i_jacobian.d_val_d_knot[i])
          * robust_wnoa::Mat3::Identity();
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian(
          jacobians[i]);
      jacobian = residual_sqrt_weight_ * sqrt_information_ * raw_jacobian;
    }
    return true;
  }

 private:
  double delta_t_;
  robust_wnoa::Mat4 blending_;
  robust_wnoa::Mat6 residual_sqrt_weight_;
  robust_wnoa::Mat6 sqrt_information_;
};

class RobustWnoaRotationFactor : public ceres::CostFunction {
 public:
  RobustWnoaRotationFactor(const Trajectory& trajectory,
                           size_t support_start,
                           double spectral_density,
                           const robust_wnoa::Mat6& residual_sqrt_weight =
                               robust_wnoa::Mat6::Identity())
      : RobustWnoaRotationFactor(
            robust_wnoa::SegmentDuration(trajectory, support_start),
            trajectory.cumu_blending_mats.at(support_start),
            spectral_density, residual_sqrt_weight) {}

  RobustWnoaRotationFactor(double delta_t,
                           const robust_wnoa::Mat4& cumulative_blending,
                           double spectral_density,
                           const robust_wnoa::Mat6& residual_sqrt_weight =
                               robust_wnoa::Mat6::Identity())
      : delta_t_(delta_t),
        cumulative_blending_(cumulative_blending),
        residual_sqrt_weight_(residual_sqrt_weight),
        sqrt_information_(robust_wnoa::BuildSqrtInformation(
            delta_t, spectral_density)) {
    set_num_residuals(6);
    for (int i = 0; i < 4; ++i)
      mutable_parameter_block_sizes()->push_back(4);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const std::pair<int, double> segment_start(0, 0.0);
    const std::pair<int, double> segment_end(0, 1.0);
    So3SplineView::JacobianStruct rotation_i_jacobian;
    So3SplineView::JacobianStruct rotation_j_jacobian;
    So3SplineView::JacobianStruct omega_i_jacobian;
    So3SplineView::JacobianStruct omega_j_jacobian;
    const bool compute_jacobians = jacobians != nullptr;

    const robust_wnoa::SO3 rotation_i = So3SplineView::EvaluateRotationNURBS(
        segment_start, cumulative_blending_, parameters,
        compute_jacobians ? &rotation_i_jacobian : nullptr);
    const robust_wnoa::SO3 rotation_j = So3SplineView::EvaluateRotationNURBS(
        segment_end, cumulative_blending_, parameters,
        compute_jacobians ? &rotation_j_jacobian : nullptr);
    const robust_wnoa::Vec3 omega_i = So3SplineView::VelocityBodyNURBS(
        segment_start, delta_t_, cumulative_blending_, parameters,
        compute_jacobians ? &omega_i_jacobian : nullptr);
    const robust_wnoa::Vec3 omega_j = So3SplineView::VelocityBodyNURBS(
        segment_end, delta_t_, cumulative_blending_, parameters,
        compute_jacobians ? &omega_j_jacobian : nullptr);

    const robust_wnoa::Vec3 phi =
        (rotation_i.inverse() * rotation_j).log();
    robust_wnoa::Mat3 right_jacobian_inverse;
    Sophus::rightJacobianInvSO3(phi, right_jacobian_inverse);
    robust_wnoa::Vec6 raw_residual;
    raw_residual.head<3>() = phi - delta_t_ * omega_i;
    raw_residual.tail<3>() =
        right_jacobian_inverse * omega_j - omega_i;
    Eigen::Map<robust_wnoa::Vec6> residual(residuals);
    residual = residual_sqrt_weight_ * sqrt_information_ * raw_residual;

    if (!compute_jacobians) return true;
    robust_wnoa::Mat3 left_jacobian_inverse;
    Sophus::leftJacobianInvSO3(phi, left_jacobian_inverse);
    const robust_wnoa::Mat3 mapped_omega_jacobian =
        MappedAngularVelocityJacobian(phi, omega_j);
    for (int i = 0; i < 4; ++i) {
      if (!jacobians[i]) continue;
      const robust_wnoa::Mat3 d_phi =
          left_jacobian_inverse * rotation_i.inverse().matrix()
          * (rotation_j_jacobian.d_val_d_knot[i]
             - rotation_i_jacobian.d_val_d_knot[i]);
      Eigen::Matrix<double, 6, 3> raw_jacobian;
      raw_jacobian.topRows<3>() =
          d_phi - delta_t_ * omega_i_jacobian.d_val_d_knot[i];
      raw_jacobian.bottomRows<3>() =
          mapped_omega_jacobian * d_phi
          + right_jacobian_inverse * omega_j_jacobian.d_val_d_knot[i]
          - omega_i_jacobian.d_val_d_knot[i];

      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jacobian(
          jacobians[i]);
      jacobian.setZero();
      jacobian.leftCols<3>() =
          residual_sqrt_weight_ * sqrt_information_ * raw_jacobian;
    }
    return true;
  }

 private:
  static robust_wnoa::Mat3 MappedAngularVelocityJacobian(
      const robust_wnoa::Vec3& phi, const robust_wnoa::Vec3& omega) {
    constexpr double epsilon = 1e-7;
    robust_wnoa::Mat3 jacobian;
    for (int column = 0; column < 3; ++column) {
      robust_wnoa::Vec3 delta = robust_wnoa::Vec3::Zero();
      delta[column] = epsilon;
      robust_wnoa::Mat3 plus;
      robust_wnoa::Mat3 minus;
      Sophus::rightJacobianInvSO3(phi + delta, plus);
      Sophus::rightJacobianInvSO3(phi - delta, minus);
      jacobian.col(column) =
          (plus * omega - minus * omega) / (2.0 * epsilon);
    }
    return jacobian;
  }

  double delta_t_;
  robust_wnoa::Mat4 cumulative_blending_;
  robust_wnoa::Mat6 residual_sqrt_weight_;
  robust_wnoa::Mat6 sqrt_information_;
};

}  // namespace analytic_derivative
}  // namespace cocolic
