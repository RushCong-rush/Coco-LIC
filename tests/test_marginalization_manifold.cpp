#include <odom/trajectory_estimator.h>

#include <cmath>
#include <iostream>

namespace {
bool CheckPriorOnlyRotation(bool register_support_first) {
  auto trajectory = std::make_shared<cocolic::Trajectory>(0.1);
  const Sophus::SO3d origin = Sophus::SO3d::exp(Eigen::Vector3d(0.5, -0.3, 0.8));
  trajectory->setKnotSO3(origin, 0);
  auto& rotation = trajectory->getKnotSO3(0);
  auto& position = trajectory->getKnotPos(0);
  Eigen::Vector4d rotation_origin = origin.unit_quaternion().coeffs();
  Eigen::Vector3d position_origin = position;

  // A retained rotation/position pair with no current measurement factors.
  auto prior = std::make_shared<MarginalizationInfo>();
  prior->m = 0;
  prior->n = 6;
  prior->keep_block_size = {4, 3};
  prior->keep_block_idx = {0, 3};
  prior->keep_block_data = {rotation_origin.data(), position_origin.data()};
  prior->linearized_jacobians = Eigen::Matrix<double, 6, 6>::Identity();
  prior->linearized_residuals = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Vector3d increment(0.12, -0.08, 0.06);
  prior->linearized_residuals.head<3>() =
      -2.0 * Sophus::SO3d::exp(increment).unit_quaternion().vec();
  const Eigen::Vector3d translation(0.3, -0.2, 0.1);
  prior->linearized_residuals.tail<3>() = -translation;
  std::vector<double*> blocks{rotation.data(), position.data()};

  cocolic::TrajectoryEstimatorOptions options;
  cocolic::TrajectoryEstimator estimator(trajectory, options);
  if (register_support_first) {
    std::vector<double*> support;
    estimator.AddControlPointsNURBS(0, support);
  }
  estimator.AddMarginalizationFactor(prior, blocks);
  if (!register_support_first) {
    // Registering translation support must not be needed to preserve SO(3).
    std::vector<double*> support;
    estimator.AddControlPointsNURBS(0, support, true);
  }
  const auto summary = estimator.Solve(50, false, 1);
  const double norm_error = std::abs(rotation.unit_quaternion().norm() - 1.0);
  const double orthogonality_error =
      (rotation.matrix().transpose() * rotation.matrix() - Eigen::Matrix3d::Identity()).norm();
  const double rotation_error =
      ((origin.inverse() * rotation).log() - increment).norm();
  const double translation_error = (position - position_origin - translation).norm();
  std::cout << "support_first=" << register_support_first
            << " local_parameters=" << summary.num_effective_parameters
            << " norm_error=" << norm_error
            << " orthogonality_error=" << orthogonality_error
            << " rotation_error=" << rotation_error
            << " translation_error=" << translation_error << '\n';
  return summary.IsSolutionUsable() && norm_error < 1e-12 &&
         orthogonality_error < 1e-12 && rotation_error < 1e-5 &&
         translation_error < 1e-5;
}
}  // namespace

int main() {
  const bool prior_only = CheckPriorOnlyRotation(false);
  const bool registered = CheckPriorOnlyRotation(true);
  return prior_only && registered ? 0 : 1;
}
