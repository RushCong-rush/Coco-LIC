#include <odom/trajectory_manager.h>
#include <iostream>

// Exercise the actual fine solve and marginalization with IMU-only and
// IMU/camera windows, without introducing a test-only production interface.
bool CheckWindows(const std::string &config_root, bool process_prior) {
  auto trajectory = std::make_shared<cocolic::Trajectory>(0.1);
  trajectory->SetSensorExtrinsics(cocolic::LiDARSensor, cocolic::ExtrinsicParam());
  trajectory->SetSensorExtrinsics(cocolic::CameraSensor, cocolic::ExtrinsicParam());
  const auto config = YAML::LoadFile(config_root + "/ct_odometry_hilti_erp_exp21.yaml");
  cocolic::TrajectoryManager manager(config, config_root, trajectory);
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
  std::cout << "prior=" << process_prior << " fine_solves=" << manager.opt_cnt << '\n';
  return manager.opt_cnt == 20;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  return CheckWindows(argv[1], false) && CheckWindows(argv[1], true) ? 0 : 1;
}
