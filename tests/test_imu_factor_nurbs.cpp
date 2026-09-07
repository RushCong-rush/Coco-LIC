#include <odom/factor/analytic_diff/trajectory_value_factor.h>
#include <spline/trajectory.h>

#include <array>
#include <iostream>
#include <stdexcept>

using Vec6 = Eigen::Matrix<double, 6, 1>;
using SO3 = Sophus::SO3d;
using Parameters = std::array<std::array<double, 4>, 10>;

namespace {
size_t columns = 0;
double max_error = 0.;

void Require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

Vec6 Evaluate(ceres::CostFunction &factor, const Parameters &parameters) {
  std::array<const double *, 10> pointers;
  for (size_t i = 0; i < pointers.size(); ++i) pointers[i] = parameters[i].data();
  Vec6 residual;
  Require(factor.Evaluate(pointers.data(), residual.data(), nullptr), "Evaluation failed");
  return residual;
}

void CheckJacobian(ceres::CostFunction &factor, Parameters parameters, bool rotations) {
  const auto &sizes = factor.parameter_block_sizes();
  std::vector<std::vector<double>> jacobians(sizes.size());
  std::vector<double *> jacobian_ptrs(sizes.size());
  std::vector<const double *> pointers(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    jacobians[i].resize(6 * sizes[i]);
    jacobian_ptrs[i] = jacobians[i].data();
    pointers[i] = parameters[i].data();
  }
  Vec6 residual;
  Require(factor.Evaluate(pointers.data(), residual.data(), jacobian_ptrs.data()),
          "Jacobian evaluation failed");
  Require((residual - Evaluate(factor, parameters)).norm() < 1e-10,
          "Residual changes when requesting Jacobians");
  const double step = 1e-6;
  for (size_t block = 0; block < sizes.size(); ++block) {
    Eigen::Map<const Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>>
        analytic(jacobians[block].data(), 6, sizes[block]);
    if (rotations && block < 4)
      Require(analytic.col(3).norm() == 0., "Expected analytic-local SO3 Jacobian");
    for (int axis = 0; axis < 3; ++axis) {
      Parameters plus = parameters, minus = parameters;
      if (rotations && block < 4) {
        const SO3 original(Eigen::Map<const Eigen::Quaterniond>(parameters[block].data()));
        const Eigen::Vector3d delta = step * Eigen::Vector3d::Unit(axis);
        Eigen::Map<Eigen::Vector4d>(plus[block].data()) =
            (original * SO3::exp(delta)).unit_quaternion().coeffs();
        Eigen::Map<Eigen::Vector4d>(minus[block].data()) =
            (original * SO3::exp(-delta)).unit_quaternion().coeffs();
      } else {
        plus[block][axis] += step;
        minus[block][axis] -= step;
      }
      const Vec6 numeric = (Evaluate(factor, plus) - Evaluate(factor, minus)) / (2 * step);
      const double error = (numeric - analytic.col(axis)).norm() /
                           std::max(1., numeric.norm());
      max_error = std::max(max_error, error);
      ++columns;
      if (error >= 5e-5) {
        std::cerr << "block=" << block << " axis=" << axis << " error=" << error << '\n';
        throw std::runtime_error("IMU Jacobian mismatch");
      }
    }
  }
}

void Check(const std::vector<int64_t> &times, double u, bool moving, double weight) {
  cocolic::Trajectory trajectory(.1);
  trajectory.knts = times;
  trajectory.InitBlendMat();
  const auto &ordinary = trajectory.blending_mats.front();
  const auto &cumulative = trajectory.cumu_blending_mats.front();
  Parameters parameters{};
  for (int i = 0; i < 4; ++i) {
    const SO3 rotation = SO3::exp(moving ? Eigen::Vector3d(.2 + .1*i, -.1 + .07*i*i, .04*i)
                                        : Eigen::Vector3d(.2, -.1, .05));
    Eigen::Map<Eigen::Vector4d>(parameters[i].data()) = rotation.unit_quaternion().coeffs();
    Eigen::Map<Eigen::Vector3d>(parameters[i+4].data()) = moving
        ? Eigen::Vector3d(.15*i*i, .1*i, .03*i*i*i) : Eigen::Vector3d(1., -2., .3);
  }
  Eigen::Map<Eigen::Vector3d>(parameters[8].data()) = Eigen::Vector3d(.013, -.008, .003);
  Eigen::Map<Eigen::Vector3d>(parameters[9].data()) = Eigen::Vector3d(.11, -.03, .08);
  cocolic::IMUData measurement;
  measurement.timestamp = times[3] + std::llround(u * (times[4] - times[3]));
  measurement.gyro = Eigen::Vector3d(.1, -.2, .3);
  measurement.accel = Eigen::Vector3d(.2, -.4, 9.7);
  u = double(measurement.timestamp - times[3]) / (times[4] - times[3]);
  const Eigen::Vector3d gravity(.3, -.2, 9.803372071);
  Vec6 information;
  information << weight, weight*2, weight*3, weight*.3, weight*.4, weight*.5;
  cocolic::analytic_derivative::IMUFactorNURBS factor(
      measurement.timestamp, measurement, gravity, information, times, {3, u}, ordinary, cumulative);
  CheckJacobian(factor, parameters, true);

  // Independent time finite differences check physical derivatives and gravity sign.
  const auto position = [&](double s) -> Eigen::Vector3d {
    const Eigen::Vector4d basis = ordinary * Eigen::Vector4d(1, s, s*s, s*s*s);
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
    const Eigen::Vector3d origin = Eigen::Map<const Eigen::Vector3d>(parameters[4].data());
    // Remove constant translation before second differences to avoid cancellation.
    for (int i = 0; i < 4; ++i)
      value += basis[i] * (Eigen::Map<const Eigen::Vector3d>(parameters[i+4].data()) - origin);
    return value;
  };
  const auto rotation = [&](double s) -> SO3 {
    const Eigen::Vector4d basis = cumulative * Eigen::Vector4d(1, s, s*s, s*s*s);
    SO3 value(Eigen::Map<const Eigen::Quaterniond>(parameters[0].data()));
    for (int i = 1; i < 4; ++i) {
      const SO3 a(Eigen::Map<const Eigen::Quaterniond>(parameters[i-1].data()));
      const SO3 b(Eigen::Map<const Eigen::Quaterniond>(parameters[i].data()));
      value *= SO3::exp(basis[i] * (a.inverse() * b).log());
    }
    return value;
  };
  const double du = 1e-4, dt = du * (times[4]-times[3]) * 1e-9;
  const Eigen::Vector3d omega = (rotation(u-du).inverse() * rotation(u+du)).log() / (2*dt);
  const Eigen::Vector3d acceleration = (position(u+du) - 2*position(u) + position(u-du)) / (dt*dt);
  Vec6 expected;
  expected.head<3>() = omega - measurement.gyro + Eigen::Map<const Eigen::Vector3d>(parameters[8].data());
  expected.tail<3>() = rotation(u).inverse() * (acceleration + gravity) - measurement.accel +
      Eigen::Map<const Eigen::Vector3d>(parameters[9].data());
  const Vec6 actual = Evaluate(factor, parameters).cwiseQuotient(information);
  if ((actual-expected).norm() >= 5e-5 * std::max(1., expected.norm()))
    std::cerr << "h_ns=" << times[4]-times[3] << " u=" << u << " moving=" << moving
              << " actual=" << actual.transpose() << " expected=" << expected.transpose() << '\n';
  Require((actual-expected).norm() < 5e-5 * std::max(1., expected.norm()),
          "IMU prediction disagrees with time derivatives");
}
}  // namespace

int main() {
  try {
    for (const std::vector<int64_t> times : {
         std::vector<int64_t>{-300000000,-200000000,-100000000,0,100000000,200000000,300000000},
         std::vector<int64_t>{-90000000,-60000000,-20000000,0,40000000,100000000,130000000},
         std::vector<int64_t>{910000000,940000000,980000000,1000000000,1005000000,1100000000,1130000000}})
      for (double u : {0., .000001, .2, .5, .8, .999999})
        for (bool moving : {false, true})
          for (double weight : {1., 100.}) Check(times, u, moving, weight);
    Parameters biases{};
    for (int i = 0; i < 4; ++i)
      Eigen::Map<Eigen::Vector3d>(biases[i].data()) = Eigen::Vector3d(.001*i, -.02*i, .003*i);
    cocolic::analytic_derivative::BiasFactor bias(1., Vec6::Constant(100.));
    CheckJacobian(bias, biases, false);
    std::cout << "IMU/bias columns=" << columns << " max_normalized_error=" << max_error << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
