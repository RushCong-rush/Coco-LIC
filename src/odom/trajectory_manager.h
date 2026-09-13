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

#pragma once
#include <odom/trajectory_estimator.h>
#include <imu/imu_state_estimator.h>
#include <spline/trajectory.h>
#include <utils/log_utils.h>

#include <odom/factor/analytic_diff/marginalization_factor.h>
#include <utils/opt_weight.h>

#include <fstream>

namespace cocolic
{

  struct TimeParam
  {
    TimeParam()
    {
      traj_active = -1;
      for (int i = 0; i < 2; ++i)
      {
        lio_imu_time[i] = -1;
        lio_imu_idx[i] = 0;

        lio_map_imu_time[i] = -1;
        lio_map_imu_idx[i] = 0;

        last_scan[i] = -1;
        cur_scan[i] = -1;
        visual_window[i] = -1;
      }
      last_bias_time = 0;
      cur_bias_time = 0;
    }

    void UpdateCurScan(int64_t scan_time_min, int64_t scan_time_max)
    {
      last_scan[0] = cur_scan[0];
      last_scan[1] = cur_scan[1];

      cur_scan[0] = scan_time_min;
      cur_scan[1] = scan_time_max;
    }

    int64_t lio_imu_time[2]; 
    int lio_imu_idx[2];      

    double lio_map_imu_time[2];
    int lio_map_imu_idx[2];

    double traj_active;     
    int64_t last_scan[2];  
    int64_t cur_scan[2];   
    double visual_window[2];

    int64_t last_bias_time;
    int64_t cur_bias_time;
  };

  class TrajectoryManager
  {
    struct TrajectoryDynamicsState
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      int64_t time_ns = -1;
      Eigen::Vector3d position = Eigen::Vector3d::Zero();
      Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
      Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
      Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
      Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
      Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero();
      bool valid = false;
    };

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<TrajectoryManager> Ptr;

    TrajectoryManager(const YAML::Node &node, const std::string &config_path, Trajectory::Ptr trajectory);

    double opt_min_t;
    int64_t opt_min_t_ns;
    int64_t opt_max_t_ns;
    bool use_lidar_scale;
    bool use_marg_;

    Eigen::Matrix<double, 6, 1> sqrt_info_;

    int opt_cnt;
    double t_opt_sum;

    bool UpdateTrajectoryWithPoseObs(const int iteration = 50);

    Eigen::aligned_vector<PoseData> absolute_datas_with_noise_;
    Eigen::aligned_vector<PoseData> relative_datas_with_noise_;

    void InitFactorInfo(
        const ExtrinsicParam &Ep_CtoI, const ExtrinsicParam &Ep_LtoI,
        const double image_feature_weight = 0,
        const Eigen::Vector3d &local_velocity_weight = Eigen::Vector3d::Zero());

    void SetTrajectory(Trajectory::Ptr trajectory) { trajectory_ = trajectory; }

    void SetSystemState(const SystemState &sys_state, double distance0);

    void SetOriginalPose(Eigen::Quaterniond q,
                         Eigen::Vector3d p = Eigen::Vector3d::Zero());

    void AddIMUData(const IMUData &data);

    void AddPoseData(const PoseData &data);

    size_t GetIMUDataSize() const { return imu_data_.size(); }

    const Eigen::aligned_vector<IMUData> &GetIMUData() const { return imu_data_; }

    void SetUpdatedLoop() { updated_loop_ = true; }

    void PropagateTrajectory(double scan_time_min, double scan_time_max, double t_add, bool non_uniform);

    void PredictTrajectory(int64_t scan_time_min, int64_t scan_time_max,
                           int64_t traj_max_time_ns, int knot_add_num, bool non_uniform);

    void UpdateLICPrior(
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs);

    void ClearLVIPrior()
    {
      lidar_marg_info = nullptr;
      lidar_prior_ctrl_id = std::make_pair(0, 0);
      lidar_marg_parameter_blocks.clear();
    }

    bool UpdateTrajectoryWithLIC(
        int lidar_iter, int64_t img_time_stamp,
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
        const Eigen::aligned_vector<Eigen::Vector3d> &pnp_3ds,
        const Eigen::aligned_vector<Eigen::Vector2d> &pnp_2ds,
        const int iteration = 50,
        bool final_lidar_iteration = true);

    void ConfigureControlPointDiagnostics(
        const std::string &output_dir,
        const std::string &query_times_path = "");

    void ConfigureObservabilityDiagnostics(const std::string &output_dir);

    void ConfigureRobustProcessDiagnostics(
        const std::string &output_dir,
        double scale_time_constant_s);

    void SetRobustProcessPriorEnabled(bool enabled)
    {
      robust_process_prior_enabled_ = enabled;
    }

    void SetRobustProcessTranslationEnabled(bool enabled)
    {
      robust_process_translation_enabled_ = enabled;
    }

    void SetRobustProcessTranslationWnoj(bool enabled)
    {
      robust_process_translation_wnoj_ = enabled;
    }

    void SetRobustProcessRotationEnabled(bool enabled)
    {
      robust_process_rotation_enabled_ = enabled;
    }

    void SetRobustProcessRotationWnoj(bool enabled)
    {
      robust_process_rotation_wnoj_ = enabled;
    }

    void SetRobustProcessProjectionDiagnosticsEnabled(bool enabled)
    {
      robust_process_projection_diagnostics_enabled_ = enabled;
    }

    void SetRobustProcessRiskScalingEnabled(bool enabled)
    {
      robust_process_risk_scaling_enabled_ = enabled;
    }

    void UpdateLiDARAttribute(double scan_time_min, double scan_time_max);

    void Log(std::string descri) const;

    const ImuStateEstimator::Ptr GetIMUStateEstimator() const
    {
      return imu_state_estimator_;
    }

    void ExtendTrajectory(int64_t max_time_ns);

    /// [nurbs]
    void ExtendTrajectory(int64_t max_time_ns, int division);

    IMUBias GetLatestBias() const
    {
      IMUBias bias;
      bias = all_imu_bias_.rbegin()->second;
      return bias;
    }

    const VPointCloud &GetMargCtrlPoint() const { return marg_ctrl_point_; }
    const VPointCloud &GetInitCtrlPoint() const { return init_ctrl_point_; }

    Eigen::Quaterniond GetGlobalFrame() const
    {
      Eigen::Vector3d z_axis = gravity_ / gravity_.norm();
      Eigen::Vector3d e_1(1, 0, 0);
      Eigen::Vector3d x_axis = e_1 - z_axis * z_axis.transpose() * e_1;
      x_axis = x_axis / x_axis.norm();
      Eigen::Matrix<double, 3, 1> y_axis =
          Eigen::SkewSymmetric<double>(z_axis) * x_axis;

      Eigen::Matrix<double, 3, 3> Rot;
      Rot.block<3, 1>(0, 0) = x_axis;
      Rot.block<3, 1>(0, 1) = y_axis;
      Rot.block<3, 1>(0, 2) = z_axis;

      Eigen::Matrix3d R_Map_To_G = Rot.inverse();
      Eigen::Quaterniond q_MtoG(R_Map_To_G);
      return q_MtoG;
    }

    bool verbose;

    const std::map<int, double> &GetFeatureInvDepths() const
    {
      return fea_id_inv_depths_;
    }

    void SetDivisionParam(int division_coarse, int division_refine)
    {
      division_coarse_ = division_coarse;
      division_refine_ = division_refine;
    }

    int GetDivision() { return division_; }

    void SetDivision(int division) { division_ = division; }

    void SetProcessCurImg(bool flag) { process_cur_img_ = flag; }
    void ClearGaussianFeedback() { gaussian_feedback_valid_ = false; }
    void SetGaussianFeedback(const PoseData& pose) {
      gaussian_feedback_pose_ = pose;
      gaussian_feedback_valid_ = true;
    }

    void SetIntrinsic(const Eigen::Matrix3d& K) { K_ = K; }
    void SetCameraGeometry(const std::shared_ptr<CameraGeometry> &camera_geometry)
    {
      camera_geometry_ = camera_geometry;
    }

  private:
    bool LocatedInFirstSegment(double cur_t) const
    {
      size_t knot_idx = trajectory_->GetCtrlIndexNURBS(cur_t * S_TO_NS) - 3;
      if (knot_idx < SplineOrder)
        return true;
      else
        return false;
    }

    void UpdateIMUInlio();

    void RemoveIMUData(int64_t t_window_min);

    void RemovePoseData(int64_t t_window_min);

    void InitTrajWithPropagation();

    void CaptureCoarseControlPoints();

    void WriteControlPointDiagnostics(
        TrajectoryEstimator &estimator,
        const ceres::Solver::Summary &summary,
        int lidar_factor_count,
        int imu_factor_count,
        int camera_factor_count,
        int prior_factor_count,
        double optimization_time_ms);

    struct ImuExcitation
    {
      int samples = 0;
      double gyro_rms = 0.0;
      double gyro_std = 0.0;
      double accel_norm_error = 0.0;
      double accel_std = 0.0;
    };

    ImuExcitation ComputeImuExcitation(const Eigen::Vector3d &gyro_bias,
                                       const Eigen::Vector3d &accel_bias) const;

    void WriteObservabilityDiagnostics(
        const ceres::Solver::Summary &summary,
        double optimization_time_ms);

    struct RobustProcessDiagnostic
    {
      bool valid = false;
      bool scale_ready = false;
      double interval_s = 0.0;
      double translation_q_hat = 0.0;
      double translation_q_used = 0.0;
      double translation_q_next = 0.0;
      double translation_nis = 0.0;
      double translation_influence = 0.0;
      double rotation_q_hat = 0.0;
      double rotation_q_used = 0.0;
      double rotation_q_next = 0.0;
      double rotation_nis = 0.0;
      double rotation_influence = 0.0;
    };

    void PrepareRobustProcessDiagnostic();

    std::vector<size_t> CurrentRobustProcessSupportStarts() const;

    double CurrentRobustProcessCostScale() const;

    int AddCurrentRobustProcessFactors(TrajectoryEstimator &estimator);

    void WriteRobustProcessDiagnostic(
        const ceres::Solver::Summary &summary,
        double optimization_time_ms);

    TrajectoryDynamicsState EvaluateTrajectoryDynamics(int64_t time_ns);

    void WriteTrajectoryDynamicsDiagnostics(
        size_t window_index, double data_start_time_s);

    void TranfromTraj4DoF(double t_min, double t_max, const Eigen::Matrix3d &R0,
                          const Eigen::Vector3d &t0, bool apply = true);

    void TranfromTraj4DoF(double t_min, double t_max,
                          const Eigen::Matrix3d &R_bef,
                          const Eigen::Vector3d &p_bef,
                          const Eigen::Matrix3d &R_aft,
                          const Eigen::Vector3d &p_aft, bool apply = true);

    PoseData original_pose_;
    PoseData gaussian_feedback_pose_;
    bool gaussian_feedback_valid_ = false;

    Eigen::aligned_vector<IMUData> imu_data_;
    Eigen::aligned_vector<PoseData> pose_data_;

    // State
    TimeParam tparam_;

    OptWeight opt_weight_;

    Eigen::Vector3d gravity_;

    Trajectory::Ptr trajectory_;

    ImuStateEstimator::Ptr imu_state_estimator_;

    std::map<int64_t, IMUBias> all_imu_bias_;

    // Marginazation info [lio system]
    MarginalizationInfo::Ptr lidar_marg_info;
    std::vector<double *> lidar_marg_parameter_blocks;
    std::pair<int, int> lidar_prior_ctrl_id;

    MarginalizationInfo::Ptr cam_marg_info;
    std::vector<double *> cam_marg_parameter_blocks;

    bool updated_loop_;

    VPointCloud marg_ctrl_point_;
    VPointCloud init_ctrl_point_;

    std::map<int, double> fea_id_inv_depths_;

    bool if_use_init_bg_;

    // variable
    int division_;
    // constant
    int division_coarse_;
    int division_refine_;

    int64_t cur_img_time_;
    bool process_cur_img_;

    Eigen::aligned_vector<Eigen::Vector3d> v_points_;
    Eigen::aligned_vector<Eigen::Vector2d> px_obss_;

    Eigen::Matrix3d K_;
    std::shared_ptr<CameraGeometry> camera_geometry_;

    std::string cp_uncertainty_output_dir_;
    std::ofstream cp_covariance_stream_;
    std::ofstream trajectory_covariance_stream_;
    std::ofstream trajectory_dynamics_stream_;
    std::ofstream optimization_window_stream_;
    size_t cp_uncertainty_window_index_ = 0;
    std::vector<double> cp_uncertainty_query_times_s_;
    size_t cp_uncertainty_query_index_ = 0;
    bool cp_uncertainty_has_query_times_ = false;
    Eigen::aligned_vector<Eigen::Vector3d> coarse_positions_;
    Eigen::aligned_vector<SO3d> coarse_rotations_;
    TrajectoryDynamicsState previous_window_end_dynamics_;

    std::string observability_output_dir_;
    std::ofstream observability_stream_;
    size_t observability_window_index_ = 0;
    CausalObservabilityDiagnostics pending_observability_;
    ImuExcitation pending_imu_excitation_;
    bool pending_observability_valid_ = false;

    std::string robust_process_output_dir_;
    std::ofstream robust_process_stream_;
    size_t robust_process_window_index_ = 0;
    double robust_process_scale_time_constant_s_ = 1.0;
    bool robust_process_scale_initialized_ = false;
    double robust_process_translation_log_q_ = 0.0;
    double robust_process_rotation_log_q_ = 0.0;
    RobustProcessDiagnostic pending_robust_process_;
    CameraRobustRisk pending_camera_risk_;
    ProcessProjectionResult pending_process_projection_;
    bool robust_process_prior_enabled_ = false;
    bool robust_process_translation_enabled_ = true;
    bool robust_process_translation_wnoj_ = false;
    bool robust_process_rotation_wnoj_ = false;
    bool robust_process_rotation_enabled_ = true;
    bool robust_process_projection_diagnostics_enabled_ = false;
    bool robust_process_risk_scaling_enabled_ = false;

  public:
    void ClearVisual()
    {
      cur_img_time_ = -1;
      process_cur_img_ = false;
      v_points_.clear();
      px_obss_.clear();
    }  
  };

} // namespace cocolic
