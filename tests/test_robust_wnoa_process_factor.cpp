#include <odom/factor/analytic_diff/robust_wnoa_process_factor.h>
#include <odom/factor/analytic_diff/process_information_projection.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

using TranslationFactor =
    cocolic::analytic_derivative::RobustWnoaTranslationFactor;
using RotationFactor =
    cocolic::analytic_derivative::RobustWnoaRotationFactor;
using SO3 = Sophus::SO3d;
using Residual = Eigen::Matrix<double, 6, 1>;

void Require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

Residual EvaluateTranslation(
    TranslationFactor &factor,
    std::array<Eigen::Vector3d, 4> &positions)
{
  std::array<double const *, 4> parameters;
  for (int i = 0; i < 4; ++i)
    parameters[i] = positions[i].data();
  Residual residual;
  Require(factor.Evaluate(parameters.data(), residual.data(), nullptr),
          "translation residual evaluation failed");
  return residual;
}

Residual EvaluateRotation(RotationFactor &factor,
                          std::array<SO3, 4> &rotations)
{
  std::array<double const *, 4> parameters;
  for (int i = 0; i < 4; ++i)
    parameters[i] = rotations[i].data();
  Residual residual;
  Require(factor.Evaluate(parameters.data(), residual.data(), nullptr),
          "rotation residual evaluation failed");
  return residual;
}

double CheckTranslationJacobians()
{
  TranslationFactor factor(
      0.025, cocolic::analytic_derivative::RdSplineView::blending_matrix_,
      0.018);
  std::array<Eigen::Vector3d, 4> positions = {
      Eigen::Vector3d(0.00, 0.02, -0.01),
      Eigen::Vector3d(0.04, 0.01, 0.00),
      Eigen::Vector3d(0.09, -0.01, 0.01),
      Eigen::Vector3d(0.15, -0.02, 0.015)};
  std::array<double const *, 4> parameters;
  std::array<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>, 4> jacobians;
  std::array<double *, 4> jacobian_data;
  for (int i = 0; i < 4; ++i)
  {
    parameters[i] = positions[i].data();
    jacobian_data[i] = jacobians[i].data();
  }
  Residual residual;
  Require(factor.Evaluate(parameters.data(), residual.data(),
                          jacobian_data.data()),
          "translation Jacobian evaluation failed");

  constexpr double step = 1e-7;
  double worst_relative_error = 0.0;
  for (int block = 0; block < 4; ++block)
  {
    for (int axis = 0; axis < 3; ++axis)
    {
      positions[block][axis] += step;
      const Residual plus = EvaluateTranslation(factor, positions);
      positions[block][axis] -= 2.0 * step;
      const Residual minus = EvaluateTranslation(factor, positions);
      positions[block][axis] += step;
      const Residual numeric = (plus - minus) / (2.0 * step);
      worst_relative_error = std::max(
          worst_relative_error,
          (jacobians[block].col(axis) - numeric).norm()
              / std::max(1.0, numeric.norm()));
    }
  }
  return worst_relative_error;
}

double CheckRotationJacobians()
{
  RotationFactor factor(
      0.025, cocolic::analytic_derivative::So3SplineView::blending_matrix_,
      0.42);
  std::array<SO3, 4> rotations = {
      SO3::exp(Eigen::Vector3d(0.08, -0.03, 0.02)),
      SO3::exp(Eigen::Vector3d(0.10, -0.01, 0.04)),
      SO3::exp(Eigen::Vector3d(0.13, 0.02, 0.03)),
      SO3::exp(Eigen::Vector3d(0.16, 0.05, 0.01))};
  std::array<double const *, 4> parameters;
  std::array<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>, 4> jacobians;
  std::array<double *, 4> jacobian_data;
  for (int i = 0; i < 4; ++i)
  {
    parameters[i] = rotations[i].data();
    jacobian_data[i] = jacobians[i].data();
  }
  Residual residual;
  Require(factor.Evaluate(parameters.data(), residual.data(),
                          jacobian_data.data()),
          "rotation Jacobian evaluation failed");

  constexpr double step = 1e-7;
  double worst_relative_error = 0.0;
  for (int block = 0; block < 4; ++block)
  {
    for (int axis = 0; axis < 3; ++axis)
    {
      Eigen::Vector3d increment = Eigen::Vector3d::Zero();
      increment[axis] = step;
      const SO3 original = rotations[block];
      rotations[block] = original * SO3::exp(increment);
      const Residual plus = EvaluateRotation(factor, rotations);
      rotations[block] = original * SO3::exp(-increment);
      const Residual minus = EvaluateRotation(factor, rotations);
      rotations[block] = original;
      const Residual numeric = (plus - minus) / (2.0 * step);
      worst_relative_error = std::max(
          worst_relative_error,
          (jacobians[block].col(axis) - numeric).norm()
              / std::max(1.0, numeric.norm()));
    }
  }
  return worst_relative_error;
}

void CheckInformationProjection()
{
  using cocolic::process_information_projection::Compute;
  Eigen::MatrixXd process_jacobian = Eigen::MatrixXd::Zero(6, 12);
  process_jacobian.leftCols<6>().setIdentity();

  const auto unobserved = Compute(
      Eigen::MatrixXd::Zero(12, 12), process_jacobian);
  Require(unobserved.success, "unobserved projection failed");
  Require((unobserved.sqrt_weight - Eigen::Matrix<double, 6, 6>::Identity())
              .norm() < 1e-12,
          "unobserved process direction was not preserved");

  const Eigen::MatrixXd equal_information =
      process_jacobian.transpose() * process_jacobian;
  const auto equal = Compute(equal_information, process_jacobian);
  Require(equal.success, "equal-information projection failed");
  Require((equal.weights - Eigen::Matrix<double, 6, 1>::Constant(0.5))
              .norm() < 1e-12,
          "equal measurement and process information did not give half weight");

  const auto strong = Compute(
      100.0 * equal_information, process_jacobian);
  Require(strong.success, "strong-information projection failed");
  Require((strong.weights -
           Eigen::Matrix<double, 6, 1>::Constant(1.0 / 101.0)).norm() < 1e-12,
          "strong measurement information was not suppressed");

  Eigen::MatrixXd coupled_measurement = Eigen::MatrixXd::Zero(6, 12);
  coupled_measurement.leftCols<6>().setIdentity();
  coupled_measurement.rightCols<6>() =
      -Eigen::Matrix<double, 6, 6>::Identity();
  const auto nuisance_coupled = Compute(
      coupled_measurement.transpose() * coupled_measurement,
      process_jacobian);
  Require(nuisance_coupled.success, "nuisance Schur projection failed");
  Require((nuisance_coupled.sqrt_weight -
           Eigen::Matrix<double, 6, 6>::Identity()).norm() < 1e-10,
          "nuisance direction was incorrectly treated as observation");
}

void CheckScaledRobustCost()
{
  ceres::CauchyLoss baseline(
      cocolic::analytic_derivative::robust_wnoa::kCauchyLossScale);
  ceres::ScaledLoss scaled(&baseline, 0.35, ceres::DO_NOT_TAKE_OWNERSHIP);
  double baseline_rho[3];
  double scaled_rho[3];
  baseline.Evaluate(12.0, baseline_rho);
  scaled.Evaluate(12.0, scaled_rho);
  for (int i = 0; i < 3; ++i)
    Require(std::abs(scaled_rho[i] - 0.35 * baseline_rho[i]) < 1e-12,
            "risk scaling changed the Cauchy loss shape");
}

}  // namespace

int main()
{
  try
  {
    const double translation_error = CheckTranslationJacobians();
    const double rotation_error = CheckRotationJacobians();
    const double worst_error = std::max(translation_error, rotation_error);
    if (worst_error >= 2e-5)
    {
      std::cerr << "worst relative Jacobian error: " << worst_error << '\n';
      throw std::runtime_error("robust WNOA process Jacobian mismatch");
    }
    CheckInformationProjection();
    CheckScaledRobustCost();
  }
  catch (const std::exception &error)
  {
    std::cerr << "robust_wnoa_process_factor_test failed: "
              << error.what() << '\n';
    return 1;
  }
  std::cout << "robust_wnoa_process_factor_test passed\n";
  return 0;
}
