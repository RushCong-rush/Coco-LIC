#include "utils/opt_weight.h"

#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    auto config = YAML::Load(R"(
gyroscope_noise_density: 0.001
accelerometer_noise_density: 0.01
gyroscope_random_walk: 0.00005
accelerometer_random_walk: 0.0003
)");
    const cocolic::OptWeight legacy(config);
    Require(!legacy.imu_noise_is_continuous, "Legacy mode changed");
    Require(std::abs(legacy.imu_info_vec[0] - 1000.) < 1e-12,
            "Legacy measurement weight changed");
    config["imu_measurement_rate_hz"] = 200.;
    const cocolic::OptWeight rate_only(config);
    Require(!rate_only.imu_noise_is_continuous, "Existing rate-only mode changed");
    config["imu_noise_is_continuous"] = true;
    const cocolic::OptWeight continuous(config);
    Require(continuous.imu_noise_is_continuous, "Continuous mode not enabled");
    Require((continuous.imu_info_vec - legacy.imu_info_vec / std::sqrt(200.)).norm() < 1e-12,
            "Measurement density scaling mismatch");
    const auto whole = continuous.ContinuousBiasCovariance(.1);
    Eigen::Matrix<double, 6, 6> split = Eigen::Matrix<double, 6, 6>::Zero();
    for (double dt : {.013, .027, .011, .049})
      split += continuous.ContinuousBiasCovariance(dt);
    Require((whole - split).norm() < 1e-20, "Bias covariance is not additive in time");
    Require(std::abs(whole(0, 0) - 2.5e-10) < 1e-20 &&
            std::abs(whole(3, 3) - 9.e-9) < 1e-20, "Bias density units mismatch");
    Require((whole - 20. * continuous.ContinuousBiasCovariance(.005)).norm() < 1e-20,
            "Uniform sample partition mismatch");
    config["imu_measurement_rate_hz"] = 400.;
    const cocolic::OptWeight faster(config);
    Require((faster.ContinuousBiasCovariance(.1) - whole).norm() == 0.,
            "Bias covariance must not depend on measurement rate");
    config.remove("imu_measurement_rate_hz");
    bool rejected = false;
    try { cocolic::OptWeight missing_rate(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    Require(rejected, "Incomplete continuous noise configuration accepted");
    std::cout << "Legacy, density scaling, and time partition checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
