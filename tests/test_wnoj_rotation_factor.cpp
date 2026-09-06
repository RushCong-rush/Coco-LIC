#include <odom/factor/analytic_diff/wnoj_rotation_factor.h>
#include <iostream>

using namespace cocolic::analytic_derivative;
using SO3 = Sophus::SO3d;
using Knots = std::array<SO3, 4>;
using Vec9 = Eigen::Matrix<double, 9, 1>;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Vec9 Evaluate(WnojRotationFactor& factor, const Knots& knots) {
  std::array<const double*, 4> parameters;
  for (int i = 0; i < 4; ++i) parameters[i] = knots[i].data();
  Vec9 residual;
  factor.Evaluate(parameters.data(), residual.data(), nullptr);
  return residual;
}

void CheckMapping(const Eigen::Vector3d& phi) {
  const Eigen::Vector3d w(0.3, -0.4, 0.2), a(-0.1, 0.2, 0.13);
  const Vec9 error = WnojRotationError(phi, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), w, a, 0.1);
  const double step = 1e-4;
  const SO3 rotation = SO3::exp(phi);
  const Eigen::Vector3d plus = (rotation * SO3::exp(step * w + 0.5 * step * step * a)).log();
  const Eigen::Vector3d minus = (rotation * SO3::exp(-step * w + 0.5 * step * step * a)).log();
  Require((error.segment<3>(3) - (plus - minus) / (2 * step)).norm() < 1e-7,
          "Local angular velocity mapping mismatch");
  Require((error.tail<3>() - (plus - 2 * phi + minus) / (step * step)).norm() < 1e-6,
          "Local angular acceleration mapping mismatch");
}

void Check(double h, const Eigen::Matrix4d& cumulative, const Eigen::Matrix4d& ordinary) {
  const double q = 0.17;
  Knots knots = {SO3::exp(Eigen::Vector3d(0.1, -0.03, 0.04)),
                 SO3::exp(Eigen::Vector3d(0.2, -0.01, 0.08)),
                 SO3::exp(Eigen::Vector3d(0.3, 0.06, -0.03)),
                 SO3::exp(Eigen::Vector3d(0.42, 0.15, 0.02))};
  WnojRotationFactor factor(h, cumulative, q);
  std::array<const double*, 4> parameters;
  std::array<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>, 4> jacobians;
  std::array<double*, 4> jacobian_data;
  for (int i = 0; i < 4; ++i) {
    parameters[i] = knots[i].data();
    jacobian_data[i] = jacobians[i].data();
  }
  for (int test = 0; test < 2; ++test) {
    if (test == 1) for (auto& knot : knots) knot = SO3();
    Vec9 residual;
    factor.Evaluate(parameters.data(), residual.data(), jacobian_data.data());
    Require((residual - Evaluate(factor, knots)).norm() < 1e-8 * std::max(1.0, residual.norm()),
            "Autodiff and double residual mismatch");
    for (int i = 0; i < 4; ++i) {
      Require(jacobians[i].allFinite() && jacobians[i].col(3).norm() == 0,
              "Invalid analytic-local Jacobian");
      const SO3 original = knots[i];
      for (int axis = 0; axis < 3; ++axis) {
        const Eigen::Vector3d delta = 1e-7 * Eigen::Vector3d::Unit(axis);
        knots[i] = original * SO3::exp(delta);
        const Vec9 plus = Evaluate(factor, knots);
        knots[i] = original * SO3::exp(-delta);
        const Vec9 minus = Evaluate(factor, knots);
        knots[i] = original;
        const Vec9 numerical = (plus - minus) / 2e-7;
        Require((numerical - jacobians[i].col(axis)).norm()
                    < 3e-6 * std::max(1.0, numerical.norm()), "Rotation Jacobian mismatch");
      }
    }
    if (test == 0) {
      // Cross-check the templated recurrence against existing spline queries.
      const SO3 ra = So3SplineView::EvaluateRotationNURBS({0, 0}, cumulative, parameters.data());
      const SO3 rb = So3SplineView::EvaluateRotationNURBS({0, 1}, cumulative, parameters.data());
      const auto omega = [&](double u) -> Eigen::Vector3d {
        return So3SplineView::VelocityBodyNURBS({0, u}, h, cumulative, parameters.data());
      };
      const double du = 1e-5;
      const Eigen::Vector3d aa = (omega(du) - omega(-du)) / (2 * du * h);
      const Eigen::Vector3d ab = (omega(1 + du) - omega(1 - du)) / (2 * du * h);
      const double energy = WnojRotationEnergy((ra.inverse() * rb).log(), omega(0), aa,
                                               omega(1), ab, h) / q;
      Require(std::abs(energy - residual.squaredNorm()) < 1e-6 * std::max(1.0, energy),
              "Spline dynamics/energy mismatch");
      const SO3 world_rotation = SO3::exp(Eigen::Vector3d(-0.3, 0.4, 0.6));
      Knots transformed = knots;
      for (auto& knot : transformed) knot = world_rotation * knot;
      Require((Evaluate(factor, transformed) - residual).norm() < 1e-8 * std::max(1.0, residual.norm()),
              "World-frame invariance failed");
      WnojRotationFactor converted(h * 1000, cumulative, q / std::pow(1000.0, 5));
      Require((Evaluate(converted, knots) - residual).norm() < 1e-8 * std::max(1.0, residual.norm()),
              "Time unit conversion mismatch");
    }
  }
  for (int degree = 0; degree <= 2; ++degree) {
    Eigen::Vector4d polynomial(0.2, 0, 0, 0);
    if (degree >= 1) polynomial[1] = 0.3 * h;
    if (degree >= 2) polynomial[2] = 0.5 * 0.8 * h * h;
    const Eigen::Vector4d controls = ordinary.transpose().fullPivLu().solve(polynomial);
    const Eigen::Vector3d axis = Eigen::Vector3d(1, 0.5, -0.4).normalized();
    for (int i = 0; i < 4; ++i) knots[i] = SO3::exp(controls[i] * axis);
    Require(Evaluate(factor, knots).norm() < 1e-6,
            "Single-axis constant angular acceleration should have zero residual");
  }
}

int main() {
  try {
    CheckMapping(Eigen::Vector3d::Zero());
    CheckMapping(Eigen::Vector3d(1e-6, -2e-6, 3e-6));
    CheckMapping(Eigen::Vector3d(0.2, -0.1, 0.3));
    CheckMapping(Eigen::Vector3d(1.2, -0.6, 0.5));
    cocolic::Trajectory trajectory(0.1);
    trajectory.knts = {-90000000, -60000000, -20000000, 0,
                       40000000, 100000000, 130000000};
    trajectory.InitBlendMat();
    for (double h : {0.01, 0.04, 0.1, 0.7}) {
      Check(h, So3SplineView::blending_matrix_, RdSplineView::blending_matrix_);
      Check(h, trajectory.cumu_blending_mats.back(), trajectory.blending_mats.back());
    }
    std::cout << "wnoj_rotation_factor_test passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
