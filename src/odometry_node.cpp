/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <ros/package.h>
#include <ros/ros.h>

#include <cstdlib>
#include <opencv2/core.hpp>

#include <odom/odometry_manager.h>

using namespace cocolic;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "cocolic");
  ros::NodeHandle nh("~");

  bool deterministic_experiment;
  int random_seed;
  nh.param<bool>("deterministic_experiment", deterministic_experiment, false);
  nh.param<int>("random_seed", random_seed, 0);
  if (deterministic_experiment) {
    std::srand(random_seed);
    cv::setRNGSeed(random_seed);
    cv::setNumThreads(1);
    Eigen::setNbThreads(1);
    ROS_INFO("Deterministic experiment mode enabled (seed=%d).", random_seed);
  }

  std::string config_path;
  nh.param<std::string>("config_path", config_path, "ct_odometry.yaml");
  ROS_INFO("Odometry load %s.", config_path.c_str());

  YAML::Node config_node = YAML::LoadFile(config_path);

  // std::string log_path = config_node["log_path"].as<std::string>();
  // FLAGS_log_dir = log_path;
  // FLAGS_colorlogtostderr = true;
  std::cout << "\n🥥 Start Coco-LIC Odometry 🥥";

  OdometryManager odom_manager(config_node, nh);
  int expected_gs_image_subscribers;
  nh.param<int>("expected_gs_image_subscribers", expected_gs_image_subscribers, 1);
  odom_manager.WaitFor3DGSSubscribers(expected_gs_image_subscribers);
  odom_manager.RunBag();

  double t_traj_max = odom_manager.SaveOdometry();
  if (t_traj_max < 0.0)
    return 2;
  std::cout << "\n✨ All Done.\n\n";

  return 0;
}
