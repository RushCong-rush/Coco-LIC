#include "utils/opt_weight.h"

#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char **argv) {
  try {
    auto config = YAML::Load(R"(
gyroscope_noise_density: 0.001
accelerometer_noise_density: 0.01
gyroscope_random_walk: 0.00005
accelerometer_random_walk: 0.0003
)");
    const cocolic::OptWeight legacy(config);
    Require(legacy.image_cost_scale == 1. && legacy.robust_process_cost_scale == 1.,
            "Default camera or GP weight changed");
    for (const char* key : {"image_cost_scale", "robust_process_cost_scale"}) {
      for (const char* value : {"0", "-1", ".inf", ".nan"}) {
        auto invalid_config = YAML::Clone(config);
        invalid_config[key] = YAML::Load(value);
        bool rejected = false;
        try { cocolic::OptWeight bad(invalid_config); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Invalid camera or GP cost scale accepted");
      }
      auto scaled_config = YAML::Clone(config);
      scaled_config[key] = 2.;
      const cocolic::OptWeight scaled(scaled_config);
      Require((scaled.imu_info_vec - legacy.imu_info_vec).norm() == 0.,
              "Camera or GP cost scale changed IMU information");
    }
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
    config["imu_measurement_cost_scale"] = 200.;
    const cocolic::OptWeight restored(config);
    Require((restored.imu_info_vec - legacy.imu_info_vec).norm() < 1e-12,
            "200 Hz / cost scale 200 must restore legacy measurement information");
    Require((restored.ContinuousBiasCovariance(.1) - whole).norm() == 0.,
            "Measurement cost scale changed bias covariance");
    Require(restored.rot_weight == continuous.rot_weight &&
            restored.pos_weight == continuous.pos_weight &&
            restored.lidar_weight == continuous.lidar_weight &&
            restored.image_weight == continuous.image_weight,
            "Measurement cost scale changed other factor weights");
    for (const char *scale : {"0", "-1", ".inf", ".nan"}) {
      config["imu_measurement_cost_scale"] = YAML::Load(scale);
      bool invalid = false;
      try { cocolic::OptWeight bad(config); }
      catch (const std::invalid_argument&) { invalid = true; }
      Require(invalid, "Invalid measurement cost scale accepted");
    }
    config["imu_measurement_cost_scale"] = 1.;
    config["imu_measurement_rate_hz"] = 400.;
    const cocolic::OptWeight faster(config);
    Require((faster.ContinuousBiasCovariance(.1) - whole).norm() == 0.,
            "Bias covariance must not depend on measurement rate");
    for (double rate : {150., 200., 400.}) {
      for (double multiplier : {.5, 1., 2.}) {
        config["imu_measurement_rate_hz"] = rate;
        config["imu_measurement_cost_scale"] = multiplier * rate;
        const cocolic::OptWeight weights(config);
        Require((weights.imu_info_vec - std::sqrt(multiplier) * legacy.imu_info_vec).norm() < 1e-10,
                "Sampling-rate / measurement-energy scaling mismatch");
        Require((weights.ContinuousBiasCovariance(.1) - whole).norm() == 0.,
                "Measurement weighting changed continuous bias covariance");
      }
    }
    Require(argc == 2, "Expected production config directory");
    for (const char *profile : {"hilti_erp_exp04", "hilti_erp_stage3",
         "hilti_erp_exp18", "hilti_erp_exp21", "hilti_lidar_only",
         "m2dgr_erp", "m3dgr_dual_erp", "oxford_spires_erp"}) {
      const auto node = YAML::LoadFile(std::string(argv[1]) + "/ct_odometry_" + profile + ".yaml");
      Require(node["imu_noise_is_continuous"].as<bool>(), "Production profile uses legacy bias model");
      Require(node["imu_measurement_rate_hz"] && node["imu_measurement_cost_scale"],
              "Production profile omits IMU rate or measurement energy");
      const cocolic::OptWeight weights(node);
      const double rate = node["imu_measurement_rate_hz"].as<double>();
      const double scale = node["imu_measurement_cost_scale"].as<double>();
      Require(weights.imu_info_vec.allFinite() && weights.imu_info_vec.minCoeff() > 0.,
              "Invalid production measurement information");
      Require(std::abs(weights.imu_info_vec[0] * weights.imu_noise.sigma_w - std::sqrt(scale / rate)) < 1e-12,
              "Production measurement whitening mismatch");
      const auto covariance = weights.ContinuousBiasCovariance(.1);
      Require(std::abs(covariance(0, 0) / weights.imu_noise.sigma_wb_2 - .1) < 1e-12 &&
              std::abs(covariance(3, 3) / weights.imu_noise.sigma_ab_2 - .1) < 1e-12,
              "Production bias covariance mismatch");
    }
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
