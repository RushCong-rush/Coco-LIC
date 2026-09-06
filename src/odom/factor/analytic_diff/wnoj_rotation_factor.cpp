#include <odom/factor/analytic_diff/wnoj_rotation_factor.h>
#include <ceres/jet.h>

namespace cocolic {
namespace analytic_derivative {
namespace {

template <typename T> using Vec3 = Eigen::Matrix<T, 3, 1>;
template <typename T> using Vec9 = Eigen::Matrix<T, 9, 1>;

Eigen::Matrix3d Whitening(double h, double q) {
  if (!(h > 0.0) || !(q > 0.0))
    throw std::invalid_argument("WNOJ interval and density must be positive");
  Eigen::Matrix3d covariance;
  covariance << 1.0 / 20, 1.0 / 8, 1.0 / 6,
                1.0 / 8, 1.0 / 3, 1.0 / 2,
                1.0 / 6, 1.0 / 2, 1;
  Eigen::Matrix3d scale = Eigen::Vector3d(1, h, h * h).asDiagonal();
  return covariance.llt().matrixL().solve(scale) / std::sqrt(q * std::pow(h, 5));
}

template <typename T>
Vec9<T> Error(const Vec3<T>& phi, const Vec3<T>& omega_a,
              const Vec3<T>& alpha_a, const Vec3<T>& omega_b,
              const Vec3<T>& alpha_b, double h) {
  using std::sqrt;
  using std::sin;
  using std::cos;
  const T s = phi.squaredNorm();
  T c, dc;
  // J_r^-1 = I + Phi/2 + c(||phi||^2) Phi^2; series avoids cancellation at zero.
  if (s < T(1e-4)) {
    c = T(1.0 / 12) + s / T(720) + s * s / T(30240)
        + s * s * s / T(1209600);
    dc = T(1.0 / 720) + s / T(15120) + s * s / T(403200);
  } else {
    const T theta = sqrt(s);
    const T sine = sin(theta / T(2));
    const T cotangent = cos(theta / T(2)) / sine;
    const T numerator = T(1) - theta * cotangent / T(2);
    c = numerator / s;
    dc = ((-cotangent / (T(4) * theta) + T(1) / (T(8) * sine * sine)) * s
          - numerator) / (s * s);
  }
  const auto mapped = [&](const Vec3<T>& v) -> Vec3<T> {
    return v + T(0.5) * phi.cross(v) + c * phi.cross(phi.cross(v));
  };
  const Vec3<T> nu = mapped(omega_b);
  const Vec3<T> eta = mapped(alpha_b) + T(0.5) * nu.cross(omega_b)
      + c * (nu.cross(phi.cross(omega_b)) + phi.cross(nu.cross(omega_b)))
      + T(2) * dc * phi.dot(nu) * phi.cross(phi.cross(omega_b));
  Vec9<T> error;
  error.template head<3>() = phi - T(h) * omega_a - T(0.5 * h * h) * alpha_a;
  error.template segment<3>(3) = nu - omega_a - T(h) * alpha_a;
  error.template tail<3>() = eta - alpha_a;
  return error;
}

template <typename T>
struct State {
  Sophus::SO3<T> rotation;
  Vec3<T> omega = Vec3<T>::Zero();
  Vec3<T> alpha = Vec3<T>::Zero();
};

template <typename T>
State<T> EvaluateState(const std::array<Sophus::SO3<T>, 4>& knots,
                       const Eigen::Matrix4d& blending, double h, double u) {
  const Eigen::Vector4d b = blending * Eigen::Vector4d(1, u, u * u, u * u * u);
  const Eigen::Vector4d db = blending * Eigen::Vector4d(0, 1, 2 * u, 3 * u * u) / h;
  const Eigen::Vector4d ddb = blending * Eigen::Vector4d(0, 0, 2, 6 * u) / (h * h);
  State<T> state;
  state.rotation = knots[0];
  for (int i = 1; i < 4; ++i) {
    const Vec3<T> delta = (knots[i - 1].inverse() * knots[i]).log();
    const Sophus::SO3<T> increment = Sophus::SO3<T>::exp(T(b[i]) * delta);
    const Vec3<T> omega = increment.inverse() * state.omega;
    const Vec3<T> step_omega = T(db[i]) * delta;
    state.alpha = increment.inverse() * state.alpha
                  + omega.cross(step_omega) + T(ddb[i]) * delta;
    state.omega = omega + step_omega;
    state.rotation *= increment;
  }
  return state;
}

template <typename T>
Vec9<T> Residual(const std::array<Sophus::SO3<T>, 4>& knots,
                 const Eigen::Matrix4d& blending, double h,
                 const Eigen::Matrix3d& whitening) {
  const auto a = EvaluateState(knots, blending, h, 0);
  const auto b = EvaluateState(knots, blending, h, 1);
  const Vec3<T> phi = (a.rotation.inverse() * b.rotation).log();
  const Vec9<T> raw = Error(phi, a.omega, a.alpha, b.omega, b.alpha, h);
  Vec9<T> residual = Vec9<T>::Zero();
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      residual.template segment<3>(3 * row) +=
          T(whitening(row, col)) * raw.template segment<3>(3 * col);
  return residual;
}

}  // namespace

Eigen::Matrix<double, 9, 1> WnojRotationError(
    const Eigen::Vector3d& phi, const Eigen::Vector3d& omega_a,
    const Eigen::Vector3d& alpha_a, const Eigen::Vector3d& omega_b,
    const Eigen::Vector3d& alpha_b, double h) {
  return Error(phi, omega_a, alpha_a, omega_b, alpha_b, h);
}

double WnojRotationEnergy(const Eigen::Vector3d& phi,
                         const Eigen::Vector3d& omega_a,
                         const Eigen::Vector3d& alpha_a,
                         const Eigen::Vector3d& omega_b,
                         const Eigen::Vector3d& alpha_b, double h) {
  const auto error = WnojRotationError(phi, omega_a, alpha_a, omega_b, alpha_b, h);
  Eigen::Matrix3d rows;
  for (int i = 0; i < 3; ++i) rows.row(i) = error.segment<3>(3 * i).transpose();
  return (Whitening(h, 1) * rows).squaredNorm();
}

WnojRotationFactor::WnojRotationFactor(const Trajectory& trajectory,
    size_t support_start, double q)
    : WnojRotationFactor(robust_wnoa::SegmentDuration(trajectory, support_start),
                         trajectory.cumu_blending_mats.at(support_start), q) {}

WnojRotationFactor::WnojRotationFactor(double h, const Eigen::Matrix4d& blending,
                                       double q)
    : h_(h), blending_(blending), whitening_(Whitening(h, q)) {
  set_num_residuals(9);
  for (int i = 0; i < 4; ++i) mutable_parameter_block_sizes()->push_back(4);
}

bool WnojRotationFactor::Evaluate(double const* const* parameters,
                                  double* residuals, double** jacobians) const {
  if (!jacobians) {
    std::array<Sophus::SO3d, 4> knots;
    for (int i = 0; i < 4; ++i) knots[i] = Eigen::Map<const Sophus::SO3d>(parameters[i]);
    Eigen::Map<Vec9<double>> residual(residuals);
    residual = Residual(knots, blending_, h_, whitening_);
    return true;
  }
  using Jet = ceres::Jet<double, 12>;
  std::array<Sophus::SO3<Jet>, 4> knots;
  for (int i = 0; i < 4; ++i) {
    Vec3<Jet> delta;
    for (int axis = 0; axis < 3; ++axis) delta[axis] = Jet(0.0, i * 3 + axis);
    const Sophus::SO3d knot = Eigen::Map<const Sophus::SO3d>(parameters[i]);
    knots[i] = knot.cast<Jet>() * Sophus::SO3<Jet>::exp(delta);
  }
  const Vec9<Jet> result = Residual(knots, blending_, h_, whitening_);
  for (int row = 0; row < 9; ++row) residuals[row] = result[row].a;
  // Match Coco's analytic right-perturbation parameterization: [J_local | 0].
  for (int i = 0; i < 4; ++i) {
    if (!jacobians[i]) continue;
    Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian(jacobians[i]);
    jacobian.setZero();
    for (int row = 0; row < 9; ++row)
      jacobian.row(row).head<3>() = result[row].v.segment<3>(i * 3).transpose();
  }
  return true;
}

}  // namespace analytic_derivative
}  // namespace cocolic
