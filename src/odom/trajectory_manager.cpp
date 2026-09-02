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

#include <odom/factor/analytic_diff/image_feature_factor.h>
#include <odom/factor/analytic_diff/trajectory_value_factor.h>
#include <odom/trajectory_manager.h>
#include <ros/assert.h>
#include <utils/log_utils.h>

#include <boost/filesystem.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
std::fstream myfile_t_ba;
namespace cocolic
{

  TrajectoryManager::TrajectoryManager(const YAML::Node &node,
                                       const std::string &config_path,
                                       Trajectory::Ptr trajectory)
      : verbose(false),
        cur_img_time_(-1),
        process_cur_img_(false),
        opt_weight_(OptWeight(node)),
        trajectory_(trajectory),
        lidar_marg_info(nullptr),
        cam_marg_info(nullptr)
  {
    std::string imu_yaml = node["imu_yaml"].as<std::string>();
    YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
    imu_state_estimator_ = std::make_shared<ImuStateEstimator>(imu_node);

    if_use_init_bg_ = imu_node["if_use_init_bg"].as<bool>();

    lidar_prior_ctrl_id = std::make_pair(0, 0);

    InitFactorInfo(trajectory_->GetSensorEP(CameraSensor),
                   trajectory_->GetSensorEP(LiDARSensor),
                   opt_weight_.image_weight, opt_weight_.local_velocity_info_vec);

    division_ = 0;
    use_marg_ = true;

    opt_cnt = 0;
    t_opt_sum = 0.0;

    v_points_.clear();
    px_obss_.clear();
  }

  void TrajectoryManager::ConfigureControlPointDiagnostics(
      const std::string &output_dir,
      const std::string &query_times_path)
  {
    if (output_dir.empty())
      return;

    cp_uncertainty_query_times_s_.clear();
    cp_uncertainty_query_index_ = 0;
    cp_uncertainty_has_query_times_ = !query_times_path.empty();
    if (cp_uncertainty_has_query_times_)
    {
      std::ifstream query_times_stream(query_times_path);
      if (!query_times_stream)
      {
        LOG(ERROR) << "Cannot open control-point diagnostic query times: "
                   << query_times_path;
        return;
      }

      std::string line;
      while (std::getline(query_times_stream, line))
      {
        std::istringstream line_stream(line);
        double timestamp_s = 0.0;
        if (line_stream >> timestamp_s && std::isfinite(timestamp_s))
          cp_uncertainty_query_times_s_.push_back(timestamp_s);
      }
      std::sort(cp_uncertainty_query_times_s_.begin(),
                cp_uncertainty_query_times_s_.end());
      cp_uncertainty_query_times_s_.erase(
          std::unique(cp_uncertainty_query_times_s_.begin(),
                      cp_uncertainty_query_times_s_.end()),
          cp_uncertainty_query_times_s_.end());
      if (cp_uncertainty_query_times_s_.empty())
      {
        LOG(ERROR) << "No query timestamps in " << query_times_path;
        return;
      }
    }

    boost::filesystem::create_directories(output_dir);
    cp_covariance_stream_.open(output_dir + "/cp_covariance.csv",
                               std::ios::out | std::ios::trunc);
    trajectory_covariance_stream_.open(
        output_dir + "/trajectory_covariance.csv",
        std::ios::out | std::ios::trunc);
    trajectory_dynamics_stream_.open(
        output_dir + "/trajectory_dynamics.csv",
        std::ios::out | std::ios::trunc);
    optimization_window_stream_.open(
        output_dir + "/optimization_windows.csv",
        std::ios::out | std::ios::trunc);
    if (!cp_covariance_stream_ || !trajectory_covariance_stream_ ||
        !trajectory_dynamics_stream_ || !optimization_window_stream_)
    {
      LOG(ERROR) << "Cannot open control-point diagnostic output: "
                 << output_dir;
      cp_covariance_stream_.close();
      trajectory_covariance_stream_.close();
      trajectory_dynamics_stream_.close();
      optimization_window_stream_.close();
      return;
    }

    cp_uncertainty_output_dir_ = output_dir;
    cp_covariance_stream_ << std::setprecision(17)
                          << "window_index,knot_index,knot_time_s,"
                             "position_x,position_y,position_z,"
                             "quaternion_x,quaternion_y,quaternion_z,quaternion_w,"
                             "position_cov_xx,position_cov_xy,position_cov_xz,"
                             "position_cov_yy,position_cov_yz,position_cov_zz,"
                             "rotation_cov_xx,rotation_cov_xy,rotation_cov_xz,"
                             "rotation_cov_yy,rotation_cov_yz,rotation_cov_zz,"
                             "coarse_to_fine_position_x,coarse_to_fine_position_y,"
                             "coarse_to_fine_position_z,coarse_to_fine_position_norm,"
                             "coarse_to_fine_rotation_x,coarse_to_fine_rotation_y,"
                             "coarse_to_fine_rotation_z,coarse_to_fine_rotation_norm\n";
    trajectory_covariance_stream_
        << std::setprecision(17)
        << "window_index,query_time_s,position_x,position_y,position_z,"
           "quaternion_x,quaternion_y,quaternion_z,quaternion_w,"
           "position_cov_xx,position_cov_xy,position_cov_xz,"
           "position_cov_yy,position_cov_yz,position_cov_zz,"
           "rotation_cov_xx,rotation_cov_xy,rotation_cov_xz,"
           "rotation_cov_yy,rotation_cov_yz,rotation_cov_zz\n";
    trajectory_dynamics_stream_
        << std::setprecision(17)
        << "window_index,origin_time_s,query_time_s,horizon_s,"
           "origin_position_x,origin_position_y,origin_position_z,"
           "origin_quaternion_x,origin_quaternion_y,origin_quaternion_z,"
           "origin_quaternion_w,"
           "origin_linear_velocity_x,origin_linear_velocity_y,"
           "origin_linear_velocity_z,"
           "origin_linear_acceleration_x,origin_linear_acceleration_y,"
           "origin_linear_acceleration_z,"
           "origin_angular_velocity_x,origin_angular_velocity_y,"
           "origin_angular_velocity_z,"
           "origin_angular_acceleration_x,origin_angular_acceleration_y,"
           "origin_angular_acceleration_z,"
           "target_position_x,target_position_y,target_position_z,"
           "target_quaternion_x,target_quaternion_y,target_quaternion_z,"
           "target_quaternion_w,"
           "target_linear_velocity_x,target_linear_velocity_y,"
           "target_linear_velocity_z,"
           "target_linear_acceleration_x,target_linear_acceleration_y,"
           "target_linear_acceleration_z,"
           "target_angular_velocity_x,target_angular_velocity_y,"
           "target_angular_velocity_z,"
           "target_angular_acceleration_x,target_angular_acceleration_y,"
           "target_angular_acceleration_z\n";
    optimization_window_stream_
        << std::setprecision(17)
        << "window_index,window_start_time_s,window_end_time_s,"
           "covariance_success,trajectory_queries,"
           "trajectory_covariance_successes,"
           "covariance_control_points,covariance_time_ms,"
           "solution_usable,initial_cost,final_cost,iterations,"
           "successful_steps,unsuccessful_steps,optimization_time_ms,"
           "lidar_factors,imu_factors,camera_factors,prior_factors,bias_factors\n";
  }

  void TrajectoryManager::ConfigureObservabilityDiagnostics(
      const std::string &output_dir)
  {
    if (output_dir.empty())
      return;

    boost::filesystem::create_directories(output_dir);
    observability_stream_.open(output_dir + "/observability_windows.csv",
                               std::ios::out | std::ios::trunc);
    if (!observability_stream_)
    {
      LOG(ERROR) << "Cannot open observability diagnostic output: "
                 << output_dir;
      return;
    }

    observability_output_dir_ = output_dir;
    observability_stream_ << std::setprecision(17)
                          << "window_index,window_start_time_s,"
                             "window_end_time_s,diagnostic_success,"
                             "diagnostic_time_ms,";
    const auto write_sensor_header = [this](const std::string &prefix) {
      observability_stream_
          << prefix << "_success," << prefix << "_factor_blocks,"
          << prefix << "_residual_count," << prefix << "_parameter_dimension,"
          << prefix << "_residual_rms," << prefix << "_raw_residual_rms,"
          << prefix << "_robust_influence_mean,"
          << prefix << "_evaluation_time_ms,"
          << prefix << "_joint_rank_fraction," << prefix << "_joint_eig_min,"
          << prefix << "_joint_eig_weakest_observable,"
          << prefix << "_joint_eig_median," << prefix << "_joint_eig_max,"
          << prefix << "_joint_condition," << prefix << "_rotation_rank_fraction,"
          << prefix << "_rotation_eig_min,"
          << prefix << "_rotation_eig_weakest_observable,"
          << prefix << "_rotation_eig_median,"
          << prefix << "_rotation_eig_max," << prefix << "_rotation_condition,"
          << prefix << "_position_rank_fraction," << prefix << "_position_eig_min,"
          << prefix << "_position_eig_weakest_observable,"
          << prefix << "_position_eig_median," << prefix << "_position_eig_max,"
          << prefix << "_position_condition,";
    };
    write_sensor_header("lidar");
    write_sensor_header("camera");
    observability_stream_
        << "imu_samples,imu_gyro_rms,imu_gyro_std,"
           "imu_accel_norm_error,imu_accel_std,"
           "coarse_to_fine_position_median,"
           "coarse_to_fine_position_max,"
           "coarse_to_fine_rotation_median,"
           "coarse_to_fine_rotation_max,solution_usable,"
           "initial_cost,final_cost,iterations,successful_steps,"
           "unsuccessful_steps,optimization_time_ms\n";
  }

  TrajectoryManager::TrajectoryDynamicsState
  TrajectoryManager::EvaluateTrajectoryDynamics(int64_t time_ns)
  {
    TrajectoryDynamicsState state;
    if (time_ns < trajectory_->knts.at(3) ||
        time_ns >= trajectory_->knts.back())
      return state;

    std::pair<int, double> su;
    trajectory_->GetIdxT(time_ns, su, true);
    if (su.first < 3 || su.first + 1 >= int(trajectory_->knts.size()))
      return state;

    const double delta_t =
        (trajectory_->knts[su.first + 1] - trajectory_->knts[su.first]) *
        NS_TO_S;
    const Eigen::Matrix4d& blending_matrix =
        trajectory_->blending_mats.at(su.first - 3);
    std::pair<int, double> control_point_query(su.first - 3, su.second);

    const SE3d pose = trajectory_->GetIMUPoseNsNURBS(time_ns);
    state.time_ns = time_ns;
    state.position = pose.translation();
    state.rotation = pose.unit_quaternion();
    state.linear_velocity = trajectory_->GetTransVelWorldNURBS(
        control_point_query, delta_t, blending_matrix);
    state.linear_acceleration = trajectory_->GetTransAccelWorldNURBS(
        control_point_query, delta_t, blending_matrix);
    state.angular_velocity =
        trajectory_->GetRotVelBodyNsNURBS(time_ns);
    state.angular_acceleration =
        trajectory_->GetRotAccelBodyNsNURBS(time_ns);
    state.valid = state.position.allFinite() &&
                  state.rotation.coeffs().allFinite() &&
                  state.linear_velocity.allFinite() &&
                  state.linear_acceleration.allFinite() &&
                  state.angular_velocity.allFinite() &&
                  state.angular_acceleration.allFinite();
    return state;
  }

  void TrajectoryManager::WriteTrajectoryDynamicsDiagnostics(
      size_t window_index, double data_start_time_s)
  {
    const int64_t current_end_time_ns = opt_max_t_ns - 1;
    const TrajectoryDynamicsState current_end =
        EvaluateTrajectoryDynamics(current_end_time_ns);
    if (previous_window_end_dynamics_.valid)
    {
      const int64_t window_duration_ns = opt_max_t_ns - opt_min_t_ns;
      for (int step = 1; step <= 4; ++step)
      {
        const int64_t query_time_ns =
            step < 4
                ? opt_min_t_ns + window_duration_ns * step / 4
                : current_end_time_ns;
        const TrajectoryDynamicsState target =
            EvaluateTrajectoryDynamics(query_time_ns);
        if (!target.valid)
          continue;

        const TrajectoryDynamicsState& origin =
            previous_window_end_dynamics_;
        const double origin_time_s =
            data_start_time_s + origin.time_ns * NS_TO_S;
        const double query_time_s =
            data_start_time_s + target.time_ns * NS_TO_S;
        const double horizon_s =
            (target.time_ns - origin.time_ns) * NS_TO_S;
        const auto write_state = [this](const TrajectoryDynamicsState& state) {
          trajectory_dynamics_stream_
              << state.position.x() << ',' << state.position.y() << ','
              << state.position.z() << ',' << state.rotation.x() << ','
              << state.rotation.y() << ',' << state.rotation.z() << ','
              << state.rotation.w() << ',' << state.linear_velocity.x() << ','
              << state.linear_velocity.y() << ','
              << state.linear_velocity.z() << ','
              << state.linear_acceleration.x() << ','
              << state.linear_acceleration.y() << ','
              << state.linear_acceleration.z() << ','
              << state.angular_velocity.x() << ','
              << state.angular_velocity.y() << ','
              << state.angular_velocity.z() << ','
              << state.angular_acceleration.x() << ','
              << state.angular_acceleration.y() << ','
              << state.angular_acceleration.z();
        };

        trajectory_dynamics_stream_
            << window_index << ',' << origin_time_s << ',' << query_time_s
            << ',' << horizon_s << ',';
        write_state(origin);
        trajectory_dynamics_stream_ << ',';
        write_state(target);
        trajectory_dynamics_stream_ << '\n';
      }
    }
    previous_window_end_dynamics_ = current_end;
  }

  void TrajectoryManager::CaptureCoarseControlPoints()
  {
    if (cp_uncertainty_output_dir_.empty() &&
        observability_output_dir_.empty())
      return;

    coarse_positions_.clear();
    coarse_rotations_.clear();
    coarse_positions_.reserve(trajectory_->numKnots());
    coarse_rotations_.reserve(trajectory_->numKnots());
    for (size_t i = 0; i < trajectory_->numKnots(); ++i)
    {
      coarse_positions_.push_back(trajectory_->getKnotPos(i));
      coarse_rotations_.push_back(trajectory_->getKnotSO3(i));
    }
  }

  TrajectoryManager::ImuExcitation TrajectoryManager::ComputeImuExcitation(
      const Eigen::Vector3d &gyro_bias,
      const Eigen::Vector3d &accel_bias) const
  {
    ImuExcitation result;
    const int begin = std::max(0, tparam_.lio_imu_idx[0]);
    const int end = std::min(static_cast<int>(imu_data_.size()),
                             tparam_.lio_imu_idx[1]);
    if (begin >= end)
      return result;

    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_mean = Eigen::Vector3d::Zero();
    double gyro_squared_sum = 0.0;
    for (int i = begin; i < end; ++i)
    {
      const Eigen::Vector3d gyro = imu_data_[i].gyro - gyro_bias;
      const Eigen::Vector3d accel = imu_data_[i].accel - accel_bias;
      gyro_mean += gyro;
      accel_mean += accel;
      gyro_squared_sum += gyro.squaredNorm();
      ++result.samples;
    }
    gyro_mean /= result.samples;
    accel_mean /= result.samples;

    double gyro_variance = 0.0;
    double accel_variance = 0.0;
    for (int i = begin; i < end; ++i)
    {
      const Eigen::Vector3d gyro = imu_data_[i].gyro - gyro_bias;
      const Eigen::Vector3d accel = imu_data_[i].accel - accel_bias;
      gyro_variance += (gyro - gyro_mean).squaredNorm();
      accel_variance += (accel - accel_mean).squaredNorm();
    }
    result.gyro_rms = std::sqrt(gyro_squared_sum / result.samples);
    result.gyro_std = std::sqrt(gyro_variance / result.samples);
    result.accel_norm_error =
        std::abs(accel_mean.norm() - gravity_.norm());
    result.accel_std = std::sqrt(accel_variance / result.samples);
    return result;
  }

  void TrajectoryManager::WriteObservabilityDiagnostics(
      const ceres::Solver::Summary &summary,
      double optimization_time_ms)
  {
    if (!pending_observability_valid_ || !observability_stream_)
      return;

    const double data_start_time_s =
        trajectory_->GetDataStartTime() * NS_TO_S;
    const double window_start_time_s =
        data_start_time_s + opt_min_t_ns * NS_TO_S;
    const double window_end_time_s =
        data_start_time_s + opt_max_t_ns * NS_TO_S;

    std::vector<double> position_corrections;
    std::vector<double> rotation_corrections;
    const size_t active_count = std::min(
        trajectory_->numKnots(), static_cast<size_t>(std::max(0, division_) + 3));
    const size_t first_active = trajectory_->numKnots() - active_count;
    for (size_t i = first_active; i < trajectory_->numKnots(); ++i)
    {
      if (i >= coarse_positions_.size() || i >= coarse_rotations_.size())
        continue;
      position_corrections.push_back(
          (trajectory_->getKnotPos(i) - coarse_positions_[i]).norm());
      rotation_corrections.push_back(
          (coarse_rotations_[i].inverse() * trajectory_->getKnotSO3(i))
              .log().norm());
    }
    const auto median = [](std::vector<double> values) {
      if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
      std::sort(values.begin(), values.end());
      const size_t middle = values.size() / 2;
      return values.size() % 2 == 0
                 ? 0.5 * (values[middle - 1] + values[middle])
                 : values[middle];
    };
    const auto maximum = [](const std::vector<double> &values) {
      return values.empty()
                 ? std::numeric_limits<double>::quiet_NaN()
                 : *std::max_element(values.begin(), values.end());
    };

    observability_stream_
        << observability_window_index_++ << ',' << window_start_time_s << ','
        << window_end_time_s << ',' << pending_observability_.success << ','
        << pending_observability_.computation_time_ms << ',';
    const auto write_spectrum = [this](const ObservabilitySpectrum &spectrum) {
      observability_stream_
          << spectrum.rank_fraction << ',' << spectrum.min_eigenvalue << ','
          << spectrum.weakest_observable_eigenvalue << ','
          << spectrum.median_eigenvalue << ',' << spectrum.max_eigenvalue << ','
          << spectrum.condition_number << ',';
    };
    const auto write_sensor = [this, &write_spectrum](
                                  const SensorObservability &sensor) {
      observability_stream_
          << sensor.success << ',' << sensor.factor_blocks << ','
          << sensor.residual_count << ',' << sensor.parameter_dimension << ','
          << sensor.residual_rms << ',' << sensor.raw_residual_rms << ','
          << sensor.robust_influence_mean << ','
          << sensor.evaluation_time_ms << ',';
      write_spectrum(sensor.joint);
      write_spectrum(sensor.rotation);
      write_spectrum(sensor.position);
    };
    write_sensor(pending_observability_.lidar);
    write_sensor(pending_observability_.camera);

    const int iterations =
        summary.num_successful_steps + summary.num_unsuccessful_steps;
    observability_stream_
        << pending_imu_excitation_.samples << ','
        << pending_imu_excitation_.gyro_rms << ','
        << pending_imu_excitation_.gyro_std << ','
        << pending_imu_excitation_.accel_norm_error << ','
        << pending_imu_excitation_.accel_std << ','
        << median(position_corrections) << ','
        << maximum(position_corrections) << ','
        << median(rotation_corrections) << ','
        << maximum(rotation_corrections) << ','
        << summary.IsSolutionUsable() << ',' << summary.initial_cost << ','
        << summary.final_cost << ',' << iterations << ','
        << summary.num_successful_steps << ','
        << summary.num_unsuccessful_steps << ',' << optimization_time_ms << '\n';
    observability_stream_.flush();
    pending_observability_valid_ = false;
  }

  void TrajectoryManager::WriteControlPointDiagnostics(
      TrajectoryEstimator &estimator,
      const ceres::Solver::Summary &summary,
      int lidar_factor_count,
      int imu_factor_count,
      int camera_factor_count,
      int prior_factor_count,
      double optimization_time_ms)
  {
    if (cp_uncertainty_output_dir_.empty())
      return;

    const size_t window_index = cp_uncertainty_window_index_++;
    const double data_start_time_s =
        trajectory_->GetDataStartTime() * NS_TO_S;
    const double window_start_time_s =
        data_start_time_s + opt_min_t_ns * NS_TO_S;
    const double window_end_time_s =
        data_start_time_s + opt_max_t_ns * NS_TO_S;
    std::vector<int64_t> trajectory_times_ns;
    if (cp_uncertainty_has_query_times_)
    {
      while (cp_uncertainty_query_index_ <
             cp_uncertainty_query_times_s_.size())
      {
        const double query_time_s =
            cp_uncertainty_query_times_s_[cp_uncertainty_query_index_];
        const int64_t query_time_ns = static_cast<int64_t>(std::llround(
            (query_time_s - data_start_time_s) * S_TO_NS));
        if (query_time_ns >= opt_max_t_ns)
          break;
        ++cp_uncertainty_query_index_;
        if (query_time_ns >= opt_min_t_ns)
          trajectory_times_ns.push_back(query_time_ns);
      }
    }
    else
    {
      trajectory_times_ns.push_back(opt_max_t_ns - 1);
    }

    ControlPointCovarianceResult covariance =
        estimator.ComputeControlPointCovariances(trajectory_times_ns);
    WriteTrajectoryDynamicsDiagnostics(window_index, data_start_time_s);
    const size_t successful_trajectory_queries = std::count_if(
        covariance.trajectory_queries.begin(),
        covariance.trajectory_queries.end(),
        [](const TrajectoryCovariance &query) { return query.success; });
    const int iterations =
        summary.num_successful_steps + summary.num_unsuccessful_steps;

    optimization_window_stream_
        << window_index << ',' << window_start_time_s << ','
        << window_end_time_s << ',' << covariance.success << ','
        << trajectory_times_ns.size() << ','
        << successful_trajectory_queries << ','
        << covariance.control_points.size() << ','
        << covariance.computation_time_ms << ',' << summary.IsSolutionUsable()
        << ',' << summary.initial_cost << ',' << summary.final_cost << ','
        << iterations << ',' << summary.num_successful_steps << ','
        << summary.num_unsuccessful_steps << ',' << optimization_time_ms << ','
        << lidar_factor_count << ',' << imu_factor_count << ','
        << camera_factor_count << ',' << prior_factor_count << ",1\n";

    for (const TrajectoryCovariance &trajectory_query :
         covariance.trajectory_queries)
    {
      if (!trajectory_query.success)
        continue;
      const Eigen::Quaterniond &q = trajectory_query.rotation;
      const Eigen::Vector3d &p = trajectory_query.position;
      const Eigen::Matrix3d &p_cov = trajectory_query.position_covariance;
      const Eigen::Matrix3d &r_cov = trajectory_query.rotation_covariance;
      const double query_time_s = data_start_time_s +
                                  trajectory_query.time_ns * NS_TO_S;
      trajectory_covariance_stream_
          << window_index << ',' << query_time_s << ',' << p.x() << ','
          << p.y() << ',' << p.z() << ',' << q.x() << ',' << q.y() << ','
          << q.z() << ',' << q.w() << ',' << p_cov(0, 0) << ','
          << p_cov(0, 1) << ',' << p_cov(0, 2) << ',' << p_cov(1, 1) << ','
          << p_cov(1, 2) << ',' << p_cov(2, 2) << ',' << r_cov(0, 0) << ','
          << r_cov(0, 1) << ',' << r_cov(0, 2) << ',' << r_cov(1, 1) << ','
          << r_cov(1, 2) << ',' << r_cov(2, 2) << '\n';
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const ControlPointCovariance &control_point :
         covariance.control_points)
    {
      const size_t i = control_point.knot_index;
      const Eigen::Vector3d &position = trajectory_->getKnotPos(i);
      const SO3d &rotation = trajectory_->getKnotSO3(i);
      const Eigen::Quaterniond quaternion = rotation.unit_quaternion();
      Eigen::Vector3d position_correction = Eigen::Vector3d::Constant(nan);
      Eigen::Vector3d rotation_correction = Eigen::Vector3d::Constant(nan);
      if (i < coarse_positions_.size() && i < coarse_rotations_.size())
      {
        position_correction = position - coarse_positions_[i];
        rotation_correction = (coarse_rotations_[i].inverse() * rotation).log();
      }
      const Eigen::Matrix3d &p_cov = control_point.position;
      const Eigen::Matrix3d &r_cov = control_point.rotation;
      const double knot_time_s =
          data_start_time_s + trajectory_->knts.at(i) * NS_TO_S;

      cp_covariance_stream_
          << window_index << ',' << i << ',' << knot_time_s << ','
          << position.x() << ',' << position.y() << ',' << position.z() << ','
          << quaternion.x() << ',' << quaternion.y() << ',' << quaternion.z()
          << ',' << quaternion.w() << ',' << p_cov(0, 0) << ','
          << p_cov(0, 1) << ',' << p_cov(0, 2) << ',' << p_cov(1, 1) << ','
          << p_cov(1, 2) << ',' << p_cov(2, 2) << ',' << r_cov(0, 0) << ','
          << r_cov(0, 1) << ',' << r_cov(0, 2) << ',' << r_cov(1, 1) << ','
          << r_cov(1, 2) << ',' << r_cov(2, 2) << ','
          << position_correction.x() << ',' << position_correction.y() << ','
          << position_correction.z() << ',' << position_correction.norm() << ','
          << rotation_correction.x() << ',' << rotation_correction.y() << ','
          << rotation_correction.z() << ',' << rotation_correction.norm()
          << '\n';
    }
    optimization_window_stream_.flush();
    cp_covariance_stream_.flush();
    trajectory_covariance_stream_.flush();
    trajectory_dynamics_stream_.flush();
  }

  void TrajectoryManager::InitFactorInfo(
      const ExtrinsicParam &Ep_CtoI, const ExtrinsicParam &Ep_LtoI,
      const double image_feature_weight,
      const Eigen::Vector3d &local_velocity_weight)
  {
    if (image_feature_weight > 1e-5)
    {
      Eigen::Matrix2d sqrt_info =
          image_feature_weight * Eigen::Matrix2d::Identity();

      analytic_derivative::ImageFeatureFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
      analytic_derivative::ImageFeatureFactor::sqrt_info = sqrt_info;

      analytic_derivative::Image3D2DFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
      analytic_derivative::Image3D2DFactor::sqrt_info = sqrt_info;

      analytic_derivative::ImageFeatureOnePoseFactor::SetParam(Ep_CtoI.so3,
                                                               Ep_CtoI.p);
      analytic_derivative::ImageFeatureOnePoseFactor::sqrt_info = sqrt_info;

      analytic_derivative::ImageDepthFactor::sqrt_info = sqrt_info;

      analytic_derivative::EpipolarFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
    }
    analytic_derivative::LoamFeatureOptMapPoseFactor::SetParam(Ep_LtoI.so3,
                                                               Ep_LtoI.p);
    analytic_derivative::RalativeLoamFeatureFactor::SetParam(Ep_LtoI.so3,
                                                             Ep_LtoI.p);
  }

  void TrajectoryManager::SetSystemState(const SystemState &sys_state, double distance0)
  {
    gravity_ = sys_state.g;

    SetOriginalPose(sys_state.q, sys_state.p);

    trajectory_->AddKntNs(0.0 * S_TO_NS);       // add knot t3   （t0、t1、t2 have been added in the constructor）
    trajectory_->AddKntNs(distance0 * S_TO_NS); // add knot t4
    trajectory_->SetMaxTimeNsNURBS(trajectory_->knts.back());

    SO3d R0(sys_state.q);
    for (size_t i = 0; i < trajectory_->numKnots(); i++)
    {
      trajectory_->setKnotSO3(R0, i);
    }
    // LOG(INFO) << "[debug numKnots] " << trajectory_->numKnots(); // 4

    tparam_.last_bias_time = trajectory_->maxTimeNsNURBS();
    tparam_.cur_bias_time = trajectory_->maxTimeNsNURBS();

    // TODO
    all_imu_bias_[tparam_.last_bias_time] = sys_state.bias;
    if (!if_use_init_bg_)
    {
      all_imu_bias_[tparam_.last_bias_time].gyro_bias = Eigen::Vector3d::Zero();
      all_imu_bias_[tparam_.last_bias_time].accel_bias = Eigen::Vector3d::Zero();
    }
  }

  void TrajectoryManager::SetOriginalPose(Eigen::Quaterniond q,
                                          Eigen::Vector3d p)
  {
    original_pose_.orientation.setQuaternion(q);
    original_pose_.position = p;
  }

  void TrajectoryManager::AddIMUData(const IMUData &data)
  {
    if (trajectory_->GetDataStartTime() < 0)
    {
      trajectory_->SetDataStartTime(data.timestamp);
    }
    imu_data_.emplace_back(data);
    imu_data_.back().timestamp -= trajectory_->GetDataStartTime();

    imu_state_estimator_->FeedIMUData(imu_data_.back());
  }

  void TrajectoryManager::AddPoseData(const PoseData &data)
  {
    pose_data_.emplace_back(data);
    pose_data_.back().timestamp -= trajectory_->GetDataStartTime();
  }

  void TrajectoryManager::RemoveIMUData(int64_t t_window_min)
  {
    if (t_window_min < 0)
      return;

    // https://stackoverflow.com/questions/991335/
    // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
    for (auto iter = imu_data_.begin(); iter != imu_data_.end();)
    {
      if (iter->timestamp < t_window_min)
      {
        iter = imu_data_.erase(iter);
      }
      else
      {
        break;
      }
    }
  }

  void TrajectoryManager::RemovePoseData(int64_t t_window_min)
  {
    if (t_window_min < 0)
      return;

    // https://stackoverflow.com/questions/991335/
    // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
    for (auto iter = pose_data_.begin(); iter != pose_data_.end();)
    {
      if (iter->timestamp < t_window_min)
      {
        iter = pose_data_.erase(iter);
      }
      else
      {
        break;
      }
    }
  }

  void TrajectoryManager::UpdateIMUInlio()
  {
    int64_t t_min = opt_min_t_ns;
    int64_t t_max = opt_max_t_ns;

    for (auto iter = imu_data_.begin(); iter != imu_data_.end(); ++iter)
    {
      if (iter->timestamp >= t_min)
      {
        if (iter->timestamp >= t_max)
        {
          continue;
        }
        tparam_.lio_imu_idx[0] = std::distance(imu_data_.begin(), iter);
        tparam_.lio_imu_time[0] = iter->timestamp;
        break;
      }
    }

    for (auto rter = imu_data_.rbegin(); rter != imu_data_.rend(); ++rter)
    {
      if (rter->timestamp < t_max)
      {
        tparam_.lio_imu_idx[1] =
            std::distance(imu_data_.begin(), rter.base()) - 1;
        tparam_.lio_imu_time[1] = rter->timestamp;
        break;
      }
    }
  }

  void TrajectoryManager::PredictTrajectory(int64_t scan_time_min, int64_t scan_time_max,
                                            int64_t traj_max_time_ns, int knot_add_num, bool non_uniform)
  {
    if (imu_data_.empty() || imu_data_.size() == 1)
    {
      // LOG(ERROR) << "[AppendWithIMUData] IMU data empty! ";
      return;
    }

    /// newly added interval：[opt_min_t_ns, opt_max_t_ns)
    opt_min_t_ns = trajectory_->maxTimeNsNURBS();

    /// extend trajectory by adding control points
    trajectory_->SetMaxTimeNsNURBS(traj_max_time_ns);
    opt_max_t_ns = trajectory_->maxTimeNsNURBS();
    SE3d last_knot = trajectory_->getLastKnot();
    trajectory_->extendKnotsTo(knot_add_num, last_knot);

    ////// color control point for visualization
    int intensity = 0;
    if (knot_add_num == 1)
    {
      intensity = 100;
    }
    else if (knot_add_num == 2)
    {
      intensity = 200;
    }
    else if (knot_add_num == 3)
    {
      intensity = 300;
    }
    else if (knot_add_num == 4)
    {
      intensity = 400;
    }
    for (int i = 0; i < knot_add_num; i++)
    {
      trajectory_->intensity_map[trajectory_->numKnots() + i] = intensity;
    }
    ////// color control point for visualization

    // LOG(INFO) << "[max_time_ns] " << opt_max_t_ns;
    // LOG(INFO) << "[numKnots aft extension] " << trajectory_->numKnots();

    tparam_.last_bias_time = tparam_.cur_bias_time;  // opt_min_t_ns
    tparam_.cur_bias_time = opt_max_t_ns;
    // LOG(INFO) << "[last_bias_time] " << tparam_.last_bias_time << " "
    //           << "[cur_bias_time] " << tparam_.cur_bias_time;
    tparam_.UpdateCurScan(scan_time_min, scan_time_max);
    UpdateIMUInlio();  // determine the imu data involved in this optimization

    /// optimization
    InitTrajWithPropagation();
  }

  void TrajectoryManager::InitTrajWithPropagation()
  {
    TrajectoryEstimatorOptions option;
    option.lock_ab = true;
    option.lock_wb = true;
    option.lock_g = true;
    option.lock_tran = false; // note
    option.show_residual_summary = verbose;
    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option, "Init Traj"));

    estimator->SetFixedIndex(3);

    // [0] prior factor
    if (true && lidar_marg_info)
    {
      estimator->AddMarginalizationFactor(lidar_marg_info,
                                          lidar_marg_parameter_blocks);
    }

    // [1] imu factor
    double *para_bg = all_imu_bias_.rbegin()->second.gyro_bias.data();
    double *para_ba = all_imu_bias_.rbegin()->second.accel_bias.data();
    for (int i = tparam_.lio_imu_idx[0]; i <= tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i),
                                                para_bg, para_ba,
                                                gravity_.data(), //(0, 0, 9.8)
                                                opt_weight_.imu_info_vec);
    }

    ceres::Solver::Summary summary = estimator->Solve(50, false);
    CaptureCoarseControlPoints();
    static int init_cnt = 0;
    init_cnt++;
    // LOG(INFO) << init_cnt << " TrajInitSolver " << summary.BriefReport();
    // LOG(INFO) << init_cnt << " TrajInit Successful/Unsuccessful steps: "
    //           << summary.num_successful_steps << "/"
    //           << summary.num_unsuccessful_steps;
  }

  bool TrajectoryManager::UpdateTrajectoryWithLIC(
      int lidar_iter, int64_t img_time_stamp,
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
      const Eigen::aligned_vector<Eigen::Vector3d> &pnp_3ds,
      const Eigen::aligned_vector<Eigen::Vector2d> &pnp_2ds,
      const int iteration,
      bool final_lidar_iteration)
  {
    if (point_corrs.empty() || imu_data_.empty() || imu_data_.size() == 1)
    {
      // LOG(WARNING) << " input empty data " << point_corrs.size() << ", "
      //              << imu_data_.size();
      return false;
    }

    // LOG(INFO) << "[point_corrs size] " << point_corrs.size();
    // LOG(INFO) << "[opt_domain]: "
    //           << "[" << opt_min_t_ns * NS_TO_S << ", " << opt_max_t_ns * NS_TO_S << ")";

    IMUBias last_bias = all_imu_bias_.rbegin()->second;
    all_imu_bias_[tparam_.cur_bias_time] = last_bias;
    std::map<int, double *> para_bg_vec;
    std::map<int, double *> para_ba_vec;
    {
      auto &bias0 = all_imu_bias_[tparam_.last_bias_time]; // bi
      para_bg_vec[0] = bias0.gyro_bias.data();
      para_ba_vec[0] = bias0.accel_bias.data();

      auto &bias1 = all_imu_bias_[tparam_.cur_bias_time]; // bj
      para_bg_vec[1] = bias1.gyro_bias.data();
      para_ba_vec[1] = bias1.accel_bias.data();
    }

    TrajectoryEstimatorOptions option;
    option.lock_ab = false;
    option.lock_wb = false;
    option.lock_g = true;
    option.show_residual_summary = verbose;
    option.collect_observability_diagnostics =
        lidar_iter == 0 && !observability_output_dir_.empty();
    if (lidar_iter == 0)
      pending_observability_valid_ = false;
    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option, "Before LIO"));

    int prior_factor_count = 0;
    int lidar_factor_count = 0;
    int imu_factor_count = 0;
    int camera_factor_count = 0;

    estimator->SetFixedIndex(3);

    // [0] prior factor
    if (true && lidar_marg_info)
    {
      estimator->AddMarginalizationFactor(lidar_marg_info,
                                          lidar_marg_parameter_blocks);
      prior_factor_count = 1;
    }

    // [1] lidar factor
    SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
    Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
    SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
    Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

    for (const auto &v : point_corrs)
    {
      if (v.t_point < opt_min_t_ns)
        continue;
      if (v.t_point >= opt_max_t_ns)
        continue;
      if (v.t_point < tparam_.last_scan[1])
        continue;
      if (use_lidar_scale)
      {
        estimator->AddLoamMeasurementAnalyticNURBS(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                                   opt_weight_.lidar_weight * v.scale);
      }
      else
      {
        estimator->AddLoamMeasurementAnalyticNURBS(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                                   opt_weight_.lidar_weight);
      }
      ++lidar_factor_count;
    }

    // [2] imu factor
    for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i), para_bg_vec[0],
                                                para_ba_vec[0], gravity_.data(),
                                                opt_weight_.imu_info_vec);
      ++imu_factor_count;
    }

    /// [3] bias factor
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> noise_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    noise_covariance.block<3, 3>(0, 0) = (opt_weight_.imu_noise.sigma_wb_discrete * opt_weight_.imu_noise.sigma_wb_discrete) * Eigen::Matrix3d::Identity();
    noise_covariance.block<3, 3>(3, 3) = (opt_weight_.imu_noise.sigma_ab_discrete * opt_weight_.imu_noise.sigma_ab_discrete) * Eigen::Matrix3d::Identity();
    for (int i = tparam_.lio_imu_idx[0] + 1; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i - 1).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      double dt = (imu_data_[i].timestamp - imu_data_[i - 1].timestamp) * NS_TO_S;
      Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
      F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
      G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt;
      G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt;
      covariance = F * covariance * F.transpose() + G * noise_covariance * G.transpose();
    }

    Eigen::Matrix<double, 6, 6> sqrt_info_mat = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse()).matrixL().transpose();
    sqrt_info_ << sqrt_info_mat(0, 0), sqrt_info_mat(1, 1), sqrt_info_mat(2, 2), sqrt_info_mat(3, 3), sqrt_info_mat(4, 4), sqrt_info_mat(5, 5);
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                             para_ba_vec[1], 1, sqrt_info_);

    /// [4] pnp factor
    v_points_.clear();
    px_obss_.clear();
    if (pnp_3ds.size() != 0)
    {
      v_points_ = pnp_3ds;
      px_obss_ = pnp_2ds;
      process_cur_img_ = true;
      cur_img_time_ = img_time_stamp;
      for (int i = 0; i < pnp_3ds.size(); i++)
      {
        estimator->AddPnPMeasurementAnalyticNURBS(
            pnp_3ds[i], pnp_2ds[i],
            img_time_stamp,
            trajectory_->GetSensorEP(CameraSensor).so3,
            trajectory_->GetSensorEP(CameraSensor).p,
            *camera_geometry_, opt_weight_.image_weight);
        ++camera_factor_count;
      }
    }
    else
    {
      process_cur_img_ = false;
    }

    if (option.collect_observability_diagnostics)
    {
      pending_observability_ =
          estimator->ComputeCausalObservabilityDiagnostics();
      pending_imu_excitation_ = ComputeImuExcitation(
          Eigen::Map<Eigen::Vector3d>(para_bg_vec[0]),
          Eigen::Map<Eigen::Vector3d>(para_ba_vec[0]));
      pending_observability_valid_ = pending_observability_.success;
    }

    TicToc t_opt;
    static int loam_cnt = 0;
    ceres::Solver::Summary summary = estimator->Solve(iteration, false);
    double opt_time = t_opt.toc();
    // LOG(INFO) << "[t_opt] " << opt_time << std::endl;
    // LOG(INFO) << "LoamSolver " << summary.BriefReport();
    // LOG(INFO) << ++loam_cnt << " UpdateLio Successful/Unsuccessful steps: "
    //           << summary.num_successful_steps << "/"
    //           << summary.num_unsuccessful_steps;

    opt_cnt++;
    t_opt_sum += opt_time;

    if (final_lidar_iteration)
    {
      WriteControlPointDiagnostics(*estimator, summary, lidar_factor_count,
                                   imu_factor_count, camera_factor_count,
                                   prior_factor_count, opt_time);
      WriteObservabilityDiagnostics(summary, opt_time);
    }

    // LOG(INFO) << "[gyro_bias_new] " << all_imu_bias_.rbegin()->second.gyro_bias.x() << " "
    //           << all_imu_bias_.rbegin()->second.gyro_bias.y() << " "
    //           << all_imu_bias_.rbegin()->second.gyro_bias.z();
    // LOG(INFO) << "[acce_bias_new] " << all_imu_bias_.rbegin()->second.accel_bias.x() << " "
    //           << all_imu_bias_.rbegin()->second.accel_bias.y() << " "
    //           << all_imu_bias_.rbegin()->second.accel_bias.z();

    return true;
  }

  void TrajectoryManager::UpdateLiDARAttribute(double scan_time_min,
                                               double scan_time_max)
  {
    if (trajectory_->maxTimeNsNURBS() > 25 * S_TO_NS)
    {
      int64_t t = trajectory_->maxTimeNsNURBS() - 15 * S_TO_NS;
      RemoveIMUData(t);
      RemovePoseData(t);
    }
  }

  void TrajectoryManager::UpdateLICPrior(
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs)
  {
    TrajectoryEstimatorOptions option;
    option.is_marg_state = true;

    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option));  // AddControlPoint

    // construct a new prior
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();

    // prepare the control points and biases to be marginalized
    int lhs_idx = trajectory_->numKnots() - 1 - division_ - 2;  // retain the last 3 control points in this optimization; remember, cubic spline is adopted
    int rhs_idx = trajectory_->numKnots() - 4;

    auto &last_bias = all_imu_bias_[tparam_.last_bias_time];  // marginalize the bias bi
    auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
    std::vector<double *> drop_param;
    for (int i = lhs_idx; i <= rhs_idx; i++)
    {
      drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
      drop_param.emplace_back(trajectory_->getKnotPos(i).data());
    }
    drop_param.emplace_back(last_bias.gyro_bias.data());
    drop_param.emplace_back(last_bias.accel_bias.data());

    // [0] prior factor marginalization
    if (lidar_marg_info)
    {
      std::vector<int> drop_set;
      for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (lidar_marg_parameter_blocks[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        MarginalizationFactor *cost_function = new MarginalizationFactor(lidar_marg_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                                                       lidar_marg_parameter_blocks, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // [1] imu factor marginalization
    for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      int64_t time_ns = imu_data_.at(i).timestamp;
      std::pair<int, double> su; // i u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);
      vec.emplace_back(last_bias.gyro_bias.data());
      vec.emplace_back(last_bias.accel_bias.data());

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        ceres::CostFunction *cost_function = new analytic_derivative::IMUFactorNURBS(
            time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec, trajectory_->knts, su,
            blending_matrix, cumulative_blending_matrix);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_IMU, cost_function, NULL,
                                                                       vec, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // [2] lidar factor marginalization
    SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
    Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
    SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
    Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();
    for (const auto &v : point_corrs)
    {
      if (v.t_point < opt_min_t_ns)
        continue;
      if (v.t_point >= opt_max_t_ns)
        continue;
      if (v.t_point < tparam_.last_scan[1])
        continue;
      int64_t time_ns = v.t_point;
      std::pair<int, double> su; // i and u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        double weight = opt_weight_.lidar_weight;
        if (use_lidar_scale)
        {
          weight *= v.scale;
        }
        ceres::CostFunction *cost_function = new analytic_derivative::LoamFeatureFactorNURBS(
            time_ns, v, su, blending_matrix, cumulative_blending_matrix,
            S_GtoM, p_GinM, S_LtoI, p_LinI, weight);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_LiDAR, cost_function, NULL,
                                                                       vec, drop_set);
        int num_residuals = cost_function->num_residuals();
        Eigen::MatrixXd residuals;
        residuals.setZero(num_residuals, 1);
        cost_function->Evaluate(vec.data(), residuals.data(), nullptr);
        double dist = (residuals / weight).norm();
        if (dist < 0.05)
        // if (dist < 0.01)
        {
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
      }
    }

    // [3] bias factor marginalization
    std::vector<double *> vec;
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(cur_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());
    vec.emplace_back(cur_bias.accel_bias.data());

    std::vector<int> drop_set;
    drop_set.emplace_back(0); // bgi
    drop_set.emplace_back(2); // bai

    analytic_derivative::BiasFactor *cost_function = new analytic_derivative::BiasFactor(1, sqrt_info_);
    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Bias, cost_function, NULL,
                                                                   vec, drop_set);
    marginalization_info->addResidualBlockInfo(residual_block_info);

    /// [4] pnp factor marginalization
    if (process_cur_img_ && v_points_.size() != 0)
    {
      int64_t time_ns = cur_img_time_;
      std::pair<int, double> su; // i和u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        Eigen::Matrix3d K;
        for (int i = 0; i < v_points_.size(); i++)
        {
          ceres::CostFunction *cost_function = new analytic_derivative::PnPFactorNURBS(
              time_ns, su,
              blending_matrix, cumulative_blending_matrix,
              v_points_[i], px_obss_[i],
              trajectory_->GetSensorEP(CameraSensor).so3,
              trajectory_->GetSensorEP(CameraSensor).p,
              *camera_geometry_, opt_weight_.image_weight);
          ceres::LossFunction *loss_function = NULL;
          loss_function = new ceres::CauchyLoss(10.0); // adopted from vins-mono
          ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Image, cost_function, loss_function,
                                                                         vec, drop_set);
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
      }
    }

    marginalization_info->preMarginalize();
    marginalization_info->marginalize();
    if (lidar_marg_info)
    {
      lidar_marg_info = nullptr;
    }
    lidar_marg_info.reset(marginalization_info);
    lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
  }

} // namespace cocolic
