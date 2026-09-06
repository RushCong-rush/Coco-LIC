#pragma once

#include <odom/factor/analytic_diff/robust_wnoa_process_factor.h>

namespace cocolic {
namespace analytic_derivative {

Eigen::Matrix<double, 9, 1> WnojRotationError(
    const Eigen::Vector3d& phi, const Eigen::Vector3d& omega_a,
    const Eigen::Vector3d& alpha_a, const Eigen::Vector3d& omega_b,
    const Eigen::Vector3d& alpha_b, double h);

double WnojRotationEnergy(const Eigen::Vector3d& phi,
                         const Eigen::Vector3d& omega_a,
                         const Eigen::Vector3d& alpha_a,
                         const Eigen::Vector3d& omega_b,
                         const Eigen::Vector3d& alpha_b, double h);

class WnojRotationFactor : public ceres::CostFunction {
 public:
  WnojRotationFactor(const Trajectory& trajectory, size_t support_start,
                     double spectral_density);
  WnojRotationFactor(double h, const Eigen::Matrix4d& cumulative_blending,
                     double spectral_density);
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

 private:
  double h_;
  Eigen::Matrix4d blending_;
  Eigen::Matrix3d whitening_;
};

}  // namespace analytic_derivative
}  // namespace cocolic
