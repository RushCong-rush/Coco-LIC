#include <odom/trajectory_manager.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

// Exercise the actual fine solve and marginalization with IMU-only and
// IMU/camera windows, without introducing a test-only production interface.
bool CheckWindows(const std::string &config_root, bool process_prior,
                  const std::string &diagnostic_dir, bool continuous_noise = false,
                  double measurement_cost_scale = 1.) {
  auto trajectory = std::make_shared<cocolic::Trajectory>(0.1);
  trajectory->SetSensorExtrinsics(cocolic::LiDARSensor, cocolic::ExtrinsicParam());
  trajectory->SetSensorExtrinsics(cocolic::CameraSensor, cocolic::ExtrinsicParam());
  auto config = YAML::LoadFile(config_root + "/ct_odometry_hilti_erp_exp21.yaml");
  config["imu_noise_is_continuous"] = continuous_noise;
  if (continuous_noise) {
    config["imu_measurement_rate_hz"] = 200.;
  } else {
    config.remove("imu_measurement_rate_hz");
  }
  config["imu_measurement_cost_scale"] = measurement_cost_scale;
  cocolic::TrajectoryManager manager(config, config_root, trajectory);
  manager.ConfigureControlPointDiagnostics(diagnostic_dir, "");
  auto camera = std::make_shared<cocolic::CameraGeometry>(
      cocolic::CameraModel::EQUIRECTANGULAR, 1024, 512);
  manager.SetCameraGeometry(camera);
  manager.SetRobustProcessPriorEnabled(process_prior);
  manager.SetRobustProcessTranslationEnabled(false);
  manager.SetRobustProcessRotationEnabled(true);
  cocolic::SystemState state;
  state.q = Eigen::Quaterniond::Identity();
  state.g = Eigen::Vector3d(0, 0, 9.81);
  manager.SetSystemState(state, 0.1);
  for (int i = 2; i <= 14; ++i)
    trajectory->AddKntNs(i * 100000000LL);
  trajectory->InitBlendMat();
  for (int i = 1; i <= 10; ++i) {
    trajectory->blending_mats.push_back(trajectory->blending_mats.front());
    trajectory->cumu_blending_mats.push_back(trajectory->cumu_blending_mats.front());
  }
  for (int i = 0; i <= 240; ++i) {
    cocolic::IMUData imu;
    imu.timestamp = i * 5000000LL;
    imu.gyro.setZero();
    imu.accel = state.g;
    manager.AddIMUData(imu);
  }
  for (int window = 1; window <= 10; ++window) {
    manager.SetDivision(1);
    trajectory->startIdx = window + 1;
    manager.PredictTrajectory(window * 100000000LL,
                              (window + 1) * 100000000LL,
                              (window + 1) * 100000000LL, 1, true);
    Eigen::aligned_vector<Eigen::Vector3d> points;
    Eigen::aligned_vector<Eigen::Vector2d> pixels;
    if (window % 3 != 0) {
      for (const Eigen::Vector3d point : {Eigen::Vector3d(0, 0, 3),
                                         Eigen::Vector3d(1, 0, 3),
                                         Eigen::Vector3d(-1, 1, 3)}) {
        Eigen::Vector2d pixel;
        if (!camera->project(point, pixel)) return false;
        points.push_back(point);
        pixels.push_back(pixel);
      }
    }
    for (int iteration = 0; iteration < 2; ++iteration) {
      if (!manager.UpdateTrajectoryWithLIC(
              iteration, window * 100000000LL + 50000000LL,
              {}, points, pixels, 8, iteration == 1)) {
        std::cerr << "Empty LiDAR matches skipped valid IMU/camera window\n";
        return false;
      }
    }
    manager.UpdateLICPrior({});
    const auto pose = trajectory->GetIMUPoseNsNURBS((window + 1) * 100000000LL - 1);
    if (!manager.sqrt_info_.allFinite() || !pose.translation().allFinite() ||
        pose.translation().norm() > 1e-5 ||
        std::abs(pose.unit_quaternion().norm() - 1.0) > 1e-12)
      return false;
  }
  std::ifstream csv(diagnostic_dir + "/optimization_windows.csv");
  std::string line, field;
  std::getline(csv, line);
  std::istringstream header(line);
  size_t imu_column = 0;
  while (std::getline(header, field, ',') && field != "imu_factors")
    ++imu_column;
  if (field != "imu_factors") return false;
  int windows = 0;
  while (std::getline(csv, line)) {
    std::istringstream row(line);
    for (size_t column = 0; column <= imu_column; ++column)
      std::getline(row, field, ',');
    if (std::stoi(field) != 20) {
      std::cerr << "Expected all 20 IMU samples in [start,end), got " << field << '\n';
      return false;
    }
    ++windows;
  }
  // Continuous noise integrates over bias-state time, not the 19 sample gaps.
  const double bias_std_time = continuous_noise ? std::sqrt(.1) : .005 * std::sqrt(19.);
  Eigen::Matrix<double, 6, 1> expected_bias_info;
  expected_bias_info.head<3>().setConstant(
      1.0 / (config["gyroscope_random_walk"].as<double>() * bias_std_time));
  expected_bias_info.tail<3>().setConstant(
      1.0 / (config["accelerometer_random_walk"].as<double>() * bias_std_time));
  if (windows != 10 || !manager.sqrt_info_.isApprox(expected_bias_info, 1e-10))
    return false;
  std::cout << "prior=" << process_prior << " fine_solves=" << manager.opt_cnt << '\n';
  return manager.opt_cnt == 20;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const auto output = boost::filesystem::temp_directory_path() /
                      boost::filesystem::unique_path("cocolic-window-%%%%-%%%%");
  const bool passed = CheckWindows(argv[1], false, (output / "off").string()) &&
                      CheckWindows(argv[1], true, (output / "rotation").string()) &&
                      CheckWindows(argv[1], false, (output / "continuous_off").string(), true) &&
                      CheckWindows(argv[1], true, (output / "continuous_rotation").string(), true) &&
                      CheckWindows(argv[1], false, (output / "gain_off").string(), true, 200.) &&
                      CheckWindows(argv[1], true, (output / "gain_rotation").string(), true, 200.);
  if (passed) boost::filesystem::remove_all(output);
  else std::cerr << "Window diagnostics: " << output << '\n';
  return passed ? 0 : 1;
}
