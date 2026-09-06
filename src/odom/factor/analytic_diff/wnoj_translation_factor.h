#pragma once

#include <odom/factor/analytic_diff/robust_wnoa_process_factor.h>

namespace cocolic {
namespace analytic_derivative {

inline double WnojTranslationEnergy(const Eigen::Vector3d& position_error,
                                   const Eigen::Vector3d& velocity_error,
                                   const Eigen::Vector3d& acceleration_error,
                                   double h) {
  if (!(h > 0.0))
    throw std::invalid_argument("WNOJ interval must be positive");
  Eigen::Matrix3d covariance;
  covariance << 1.0 / 20, 1.0 / 8, 1.0 / 6,
                1.0 / 8, 1.0 / 3, 1.0 / 2,
                1.0 / 6, 1.0 / 2, 1;
  Eigen::Matrix3d error;
  error.row(0) = position_error.transpose();
  error.row(1) = h * velocity_error.transpose();
  error.row(2) = h * h * acceleration_error.transpose();
  const Eigen::Matrix3d whitened = covariance.llt().matrixL().solve(error);
  return whitened.squaredNorm() / std::pow(h, 5);
}

class WnojTranslationFactor : public ceres::CostFunction {
 public:
  WnojTranslationFactor(const Trajectory& trajectory, size_t support_start,
                        double spectral_density)
      : WnojTranslationFactor(
            robust_wnoa::SegmentDuration(trajectory, support_start),
            trajectory.blending_mats.at(support_start), spectral_density) {}

  WnojTranslationFactor(double h, const Eigen::Matrix4d& blending,
                        double spectral_density) {
    if (!(h > 0.0) || !(spectral_density > 0.0))
      throw std::invalid_argument("WNOJ interval and density must be positive");
    // On one cubic segment, e^T Q_J^-1 e = h * ||jerk||^2 / q_J.
    coefficients_ = (6.0 * std::sqrt(h / spectral_density) / (h * h * h))
                    * blending.col(3);
    set_num_residuals(3);
    for (int i = 0; i < 4; ++i)
      mutable_parameter_block_sizes()->push_back(3);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual.setZero();
    for (int i = 0; i < 4; ++i) {
      residual += coefficients_[i] * Eigen::Map<const Eigen::Vector3d>(parameters[i]);
      if (jacobians && jacobians[i]) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[i]);
        jacobian = coefficients_[i] * Eigen::Matrix3d::Identity();
      }
    }
    return true;
  }

 private:
  Eigen::Vector4d coefficients_;
};

}  // namespace analytic_derivative
}  // namespace cocolic
