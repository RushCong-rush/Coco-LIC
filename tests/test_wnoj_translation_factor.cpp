#include <odom/factor/analytic_diff/wnoj_translation_factor.h>

#include <iostream>

using namespace cocolic::analytic_derivative;
using Points = std::array<Eigen::Vector3d, 4>;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Eigen::Vector3d Evaluate(WnojTranslationFactor& factor, const Points& points) {
  std::array<const double*, 4> parameters;
  for (int i = 0; i < 4; ++i) parameters[i] = points[i].data();
  Eigen::Vector3d residual;
  factor.Evaluate(parameters.data(), residual.data(), nullptr);
  return residual;
}

void Check(double h, const Eigen::Matrix4d& blending) {
  const double q = 0.017;
  Points points = {Eigen::Vector3d(0, 0.03, -0.1),
                   Eigen::Vector3d(0.1, -0.02, 0.02),
                   Eigen::Vector3d(0.21, 0.01, -0.02),
                   Eigen::Vector3d(0.32, 0.07, 0.03)};
  WnojTranslationFactor factor(h, blending, q);
  std::array<const double*, 4> parameters;
  std::array<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>, 4> jacobians;
  std::array<double*, 4> jacobian_data;
  for (int i = 0; i < 4; ++i) {
    parameters[i] = points[i].data();
    jacobian_data[i] = jacobians[i].data();
  }
  Eigen::Vector3d residual;
  factor.Evaluate(parameters.data(), residual.data(), jacobian_data.data());
  for (int i = 0; i < 4; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const double old = points[i][axis];
      points[i][axis] = old + 1e-7;
      const Eigen::Vector3d plus = Evaluate(factor, points);
      points[i][axis] = old - 1e-7;
      const Eigen::Vector3d minus = Evaluate(factor, points);
      points[i][axis] = old;
      const Eigen::Vector3d numerical = (plus - minus) / 2e-7;
      Require((numerical - jacobians[i].col(axis)).norm()
                  < 1e-6 * std::max(1.0, numerical.norm()), "Jacobian mismatch");
    }
  }
  const Eigen::Vector3d p0 = RdSplineView::evaluateNURBS({0, 0}, blending, parameters.data());
  const Eigen::Vector3d p1 = RdSplineView::evaluateNURBS({0, 1}, blending, parameters.data());
  const Eigen::Vector3d v0 = RdSplineView::velocityNURBS({0, 0}, h, blending, parameters.data());
  const Eigen::Vector3d v1 = RdSplineView::velocityNURBS({0, 1}, h, blending, parameters.data());
  const Eigen::Vector3d a0 = RdSplineView::evaluateNURBS<2>({0, 0}, blending, parameters.data()) / (h * h);
  const Eigen::Vector3d a1 = RdSplineView::evaluateNURBS<2>({0, 1}, blending, parameters.data()) / (h * h);
  const double full_energy = WnojTranslationEnergy(
      p1 - p0 - h * v0 - 0.5 * h * h * a0, v1 - v0 - h * a0, a1 - a0, h) / q;
  Require(std::abs(full_energy - residual.squaredNorm())
              < 1e-9 * std::max(1.0, full_energy), "Reduced/full energy mismatch");

  // Units: p' = length_scale * p, t' = time_scale * t, q' = L^2/T^5 * q.
  const double length_scale = 1000, time_scale = 1000;
  Points scaled = points;
  for (auto& point : scaled) point *= length_scale;
  WnojTranslationFactor converted(h * time_scale, blending,
      q * length_scale * length_scale / std::pow(time_scale, 5));
  Require((Evaluate(converted, scaled) - residual).norm()
              < 1e-9 * std::max(1.0, residual.norm()), "Unit conversion mismatch");

  for (int degree = 0; degree <= 2; ++degree) {
    Eigen::Vector4d polynomial = Eigen::Vector4d::Zero();
    polynomial[0] = 1.2;
    if (degree >= 1) polynomial[1] = -0.3 * h;
    if (degree >= 2) polynomial[2] = 0.5 * 0.8 * h * h;
    const Eigen::Vector4d controls = blending.transpose().fullPivLu().solve(polynomial);
    for (int i = 0; i < 4; ++i) points[i] = controls[i] * Eigen::Vector3d(1, 0.5, -0.4);
    Require(Evaluate(factor, points).norm() < 1e-7,
            "Constant position/velocity/acceleration should have zero residual");
  }
}

int main() {
  try {
    cocolic::Trajectory trajectory(0.1);
    trajectory.knts = {-90000000, -60000000, -20000000, 0,
                       40000000, 100000000, 130000000};
    trajectory.InitBlendMat();
    for (double h : {0.01, 0.04, 0.1, 0.7}) {
      Check(h, RdSplineView::blending_matrix_);
      Check(h, trajectory.blending_mats.back());
    }
    bool rejected = false;
    try { WnojTranslationFactor invalid(0, Eigen::Matrix4d::Identity(), 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    Require(rejected, "Non-positive interval was accepted");
    std::cout << "wnoj_translation_factor_test passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
