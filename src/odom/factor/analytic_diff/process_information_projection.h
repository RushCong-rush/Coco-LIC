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

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <limits>

namespace cocolic {
namespace process_information_projection {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

struct Result {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool success = false;
  int state_dimension = 0;
  int process_rank = 0;
  Mat6 sqrt_weight = Mat6::Zero();
  Vec6 relative_information = Vec6::Zero();
  Vec6 weights = Vec6::Zero();
};

inline Eigen::MatrixXd SymmetricPseudoInverse(
    const Eigen::MatrixXd& matrix) {
  if (matrix.rows() == 0)
    return Eigen::MatrixXd(0, 0);

  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if (solver.info() != Eigen::Success)
    return Eigen::MatrixXd();

  const Eigen::VectorXd eigenvalues = solver.eigenvalues();
  const double largest = std::max(0.0, eigenvalues.maxCoeff());
  const double threshold = largest * matrix.rows()
      * std::numeric_limits<double>::epsilon();
  Eigen::VectorXd inverse = Eigen::VectorXd::Zero(eigenvalues.size());
  for (int i = 0; i < eigenvalues.size(); ++i) {
    if (eigenvalues[i] > threshold)
      inverse[i] = 1.0 / eigenvalues[i];
  }
  return solver.eigenvectors() * inverse.asDiagonal()
      * solver.eigenvectors().transpose();
}

inline Result Compute(const Eigen::MatrixXd& baseline_information,
                      const Eigen::MatrixXd& process_jacobian) {
  Result result;
  const int state_dimension = baseline_information.rows();
  result.state_dimension = state_dimension;
  if (state_dimension < 6 ||
      baseline_information.cols() != state_dimension ||
      process_jacobian.rows() != 6 ||
      process_jacobian.cols() != state_dimension ||
      !baseline_information.allFinite() || !process_jacobian.allFinite())
    return result;

  Eigen::JacobiSVD<Eigen::MatrixXd> process_svd(
      process_jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::VectorXd singular_values = process_svd.singularValues();
  const double largest_singular = singular_values[0];
  const double rank_threshold = largest_singular
      * std::max(process_jacobian.rows(), process_jacobian.cols())
      * std::numeric_limits<double>::epsilon();
  result.process_rank =
      (singular_values.array() > rank_threshold).count();
  if (result.process_rank != 6)
    return result;

  const Eigen::MatrixXd information =
      0.5 * (baseline_information + baseline_information.transpose());
  const Eigen::MatrixXd process_basis = process_svd.matrixV().leftCols(6);
  const int nuisance_dimension = state_dimension - 6;
  Eigen::MatrixXd conditional_information =
      process_basis.transpose() * information * process_basis;
  if (nuisance_dimension > 0) {
    const Eigen::MatrixXd nuisance_basis =
        process_svd.matrixV().rightCols(nuisance_dimension);
    const Eigen::MatrixXd cross_information =
        process_basis.transpose() * information * nuisance_basis;
    const Eigen::MatrixXd nuisance_information =
        nuisance_basis.transpose() * information * nuisance_basis;
    const Eigen::MatrixXd nuisance_inverse =
        SymmetricPseudoInverse(nuisance_information);
    if (nuisance_inverse.rows() != nuisance_dimension)
      return result;
    conditional_information -= cross_information * nuisance_inverse
        * cross_information.transpose();
  }
  conditional_information = 0.5 *
      (conditional_information + conditional_information.transpose());

  const Eigen::Matrix<double, 6, 6> inverse_singular =
      singular_values.cwiseInverse().asDiagonal();
  Eigen::Matrix<double, 6, 6> relative_information =
      inverse_singular * conditional_information * inverse_singular;
  relative_information = 0.5 *
      (relative_information + relative_information.transpose());
  Eigen::SelfAdjointEigenSolver<Mat6> relative_solver(relative_information);
  if (relative_solver.info() != Eigen::Success)
    return result;

  result.relative_information =
      relative_solver.eigenvalues().cwiseMax(0.0);
  result.weights =
      (Vec6::Ones() + result.relative_information).cwiseInverse();
  const Mat6 sqrt_weight_in_process_basis =
      relative_solver.eigenvectors()
      * result.weights.cwiseSqrt().asDiagonal()
      * relative_solver.eigenvectors().transpose();
  result.sqrt_weight = process_svd.matrixU()
      * sqrt_weight_in_process_basis * process_svd.matrixU().transpose();
  result.success = result.sqrt_weight.allFinite();
  return result;
}

}  // namespace process_information_projection
}  // namespace cocolic
