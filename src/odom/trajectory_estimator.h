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

#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <odom/factor/ceres_local_param.h>
#include <imu/imu_state_estimator.h>
#include <lidar/lidar_feature.h>
#include <utils/parameter_struct.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <limits>

#include "opt_param.h"
#include "trajectory_estimator_options.h"

#include <odom/factor/analytic_diff/image_feature_factor.h>
#include <odom/factor/analytic_diff/lidar_feature_factor.h>
#include <odom/factor/analytic_diff/marginalization_factor.h>
#include <odom/factor/analytic_diff/process_information_projection.h>
#include <odom/factor/analytic_diff/robust_wnoa_process_factor.h>
#include <odom/factor/analytic_diff/wnoj_translation_factor.h>
#include <odom/factor/analytic_diff/wnoj_rotation_factor.h>
#include <odom/factor/analytic_diff/trajectory_value_factor.h>

namespace cocolic
{

  struct ControlPointCovariance
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    size_t knot_index = 0;
    Eigen::Matrix3d position = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
  };

  struct TrajectoryCovariance
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool success = false;
    int64_t time_ns = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Zero();
  };

  struct ControlPointCovarianceResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool success = false;
    double computation_time_ms = 0.0;
    Eigen::aligned_vector<ControlPointCovariance> control_points;
    Eigen::aligned_vector<TrajectoryCovariance> trajectory_queries;
  };

  struct ObservabilitySpectrum
  {
    int dimension = 0;
    int effective_rank = 0;
    double rank_fraction = 0.0;
    double min_eigenvalue = 0.0;
    double weakest_observable_eigenvalue = 0.0;
    double median_eigenvalue = 0.0;
    double max_eigenvalue = 0.0;
    double condition_number = std::numeric_limits<double>::infinity();
  };

  struct SensorObservability
  {
    bool success = false;
    int factor_blocks = 0;
    int residual_count = 0;
    int parameter_dimension = 0;
    double residual_rms = std::numeric_limits<double>::quiet_NaN();
    double raw_residual_rms = std::numeric_limits<double>::quiet_NaN();
    double robust_influence_mean = std::numeric_limits<double>::quiet_NaN();
    double evaluation_time_ms = 0.0;
    ObservabilitySpectrum joint;
    ObservabilitySpectrum rotation;
    ObservabilitySpectrum position;
  };

  struct CausalObservabilityDiagnostics
  {
    bool success = false;
    double computation_time_ms = 0.0;
    SensorObservability lidar;
    SensorObservability camera;
  };

  struct CameraRobustRisk
  {
    bool valid = false;
    int factor_blocks = 0;
    double influence_mean = std::numeric_limits<double>::quiet_NaN();
    double risk = 0.0;
    double computation_time_ms = 0.0;
  };

  struct ProcessProjectionChannel
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool success = false;
    int process_rank = 0;
    Eigen::Matrix<double, 6, 6> sqrt_weight =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> relative_information =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> weights =
        Eigen::Matrix<double, 6, 1>::Zero();
  };

  struct ProcessProjection
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    size_t support_start = 0;
    ProcessProjectionChannel translation;
    ProcessProjectionChannel rotation;
  };

  struct ProcessProjectionResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool success = false;
    int residual_count = 0;
    int state_dimension = 0;
    double computation_time_ms = 0.0;
    Eigen::aligned_vector<ProcessProjection> projections;
  };

  struct ResidualSummary
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::map<ResidualType, std::vector<double>> err_type_sum;
    std::map<ResidualType, int> err_type_number;

    std::map<ResidualType, time_span_t> err_type_duration;
    std::map<int, std::pair<size_t, size_t>> opt_knot;

    std::string descri_info;

    ResidualSummary(std::string descri = "") : descri_info(descri)
    {
      for (auto typ = RType_Pose; typ <= RType_Prior;
           typ = ResidualType(typ + 1))
      {
        err_type_sum[typ].push_back(0);
        err_type_number[typ] = 0;
        err_type_duration[typ] = std::make_pair(0, 0);
      }
      opt_knot[0] = std::make_pair(1, 0); // pos knot
      opt_knot[1] = std::make_pair(1, 0); // rot knot
    }

    void AddResidualInfo(ResidualType r_type,
                         const ceres::CostFunction *cost_function,
                         const std::vector<double *> &param_vec);

    void AddResidualTimestamp(ResidualType r_type, int64_t time_ns)
    {
      auto &t_span = err_type_duration[r_type];
      if (t_span.first == 0)
      {
        t_span.first = time_ns;
        t_span.second = time_ns;
      }
      else
      {
        t_span.first = t_span.first < time_ns ? t_span.first : time_ns;
        t_span.second = t_span.second > time_ns ? t_span.second : time_ns;
      }
    }

    void AddKnotIdx(size_t knot, bool is_pos_knot)
    {
      int k = is_pos_knot ? 0 : 1;
      if (opt_knot[k].first > opt_knot[k].second)
      {
        opt_knot[k].first = knot;
        opt_knot[k].second = knot;
      }
      else
      {
        opt_knot[k].first = opt_knot[k].first < knot ? opt_knot[k].first : knot;
        opt_knot[k].second =
            opt_knot[k].second > knot ? opt_knot[k].second : knot;
      }
    }

    void PrintSummary(int64_t t0_ns, int64_t dt_ns,
                      int fixed_ctrl_idx = -1) const;

    std::string GetTimeString(int64_t knot_min, int64_t knot_max, int64_t t0_ns,
                              int64_t dt_ns) const;

    std::string GetCtrlString(int64_t t_min_ns, int64_t t_max_ns, int64_t t0_ns,
                              int64_t dt_ns) const;
  };

  class TrajectoryEstimator
  {
    static ceres::Problem::Options DefaultProblemOptions()
    {
      ceres::Problem::Options options;
      options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
      options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      return options;
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<TrajectoryEstimator> Ptr;

    TrajectoryEstimator(Trajectory::Ptr trajectory,
                        TrajectoryEstimatorOptions &option,
                        std::string descri = "");

    ~TrajectoryEstimator()
    {
      if (analytic_local_parameterization_)
        delete analytic_local_parameterization_;

      if (auto_diff_local_parameterization_)
        delete auto_diff_local_parameterization_;

      if (homo_vec_local_parameterization_)
        delete homo_vec_local_parameterization_;

      // delete marginalization_info_;
    }

    void SetKeyScanConstant(double max_time);

    bool MeasuredTimeToNs(const SensorType &sensor_type, const double &timestamp,
                          int64_t &time_ns) const;

    void SetFixedIndex(int idx) { fixed_control_point_index_ = idx; }

    int GetFixedControlIndex() const { return fixed_control_point_index_; }

    void SetTimeoffsetState();

    void AddStartTimePose(const PoseData &pose);

    void AddPoseMeasurementAnalytic(const PoseData &pose_data,
                                    const Eigen::Matrix<double, 6, 1> &info_vec);

    void AddIMUMeasurementAnalytic(const IMUData &imu_data, double *gyro_bias,
                                   double *accel_bias, double *gravity,
                                   const Eigen::Matrix<double, 6, 1> &info_vec,
                                   bool marg_this_factor = false);
    void AddIMUMeasurementAnalyticNURBS(const IMUData &imu_data, double *gyro_bias,
                                        double *accel_bias, double *gravity,
                                        const Eigen::Matrix<double, 6, 1> &info_vec,
                                        bool marg_this_factor = false);

    void AddBiasFactor(double *bias_gyr_i, double *bias_gyr_j, double *bias_acc_i,
                       double *bias_acc_j, double dt,
                       const Eigen::Matrix<double, 6, 1> &info_vec,
                       bool marg_this_factor = false, bool marg_all_bias = false);

    void AddGravityFactor(double *gravity, const Eigen::Vector3d &info_vec,
                          bool marg_this_factor = false);

    void AddRelativeRotationAnalytic(double ta, double tb, const SO3d &S_BtoA,
                                     const Eigen::Vector3d &info_vec);

    bool AddGlobalVelocityMeasurement(const double timestamp,
                                      const Eigen::Vector3d &global_v,
                                      double vel_weight);

    void AddLocalVelocityMeasurementAnalytic(const double timestamp,
                                             const Eigen::Vector3d &local_v,
                                             double weight);

    void AddLocal6DoFVelocityAnalytic(
        const double timestamp, const Eigen::Matrix<double, 6, 1> &local_v,
        const Eigen::Matrix<double, 6, 1> &info_vec);

    void AddLoamMeasurementAnalytic(const PointCorrespondence &pc,
                                    const SO3d &S_GtoM,
                                    const Eigen::Vector3d &p_GinM,
                                    const SO3d &S_LtoI,
                                    const Eigen::Vector3d &p_LinI, double weight,
                                    bool marg_this_factor = false);

    void AddLoamMeasurementAnalyticNURBS(const PointCorrespondence &pc,
                                         const SO3d &S_GtoM,
                                         const Eigen::Vector3d &p_GinM,
                                         const SO3d &S_LtoI,
                                         const Eigen::Vector3d &p_LinI, double weight,
                                         bool marg_this_factor = false);

    void AddPhotometricMeasurementAnalyticNURBS(const double &prev_pixel_intensity,
                                                const Eigen::Vector3d &visual_map_point,
                                                const cv::Mat &cur_img, int64_t cur_img_timestamp,
                                                const SO3d &S_VtoI, const Eigen::Vector3d &p_VinI, const Eigen::Matrix3d &K,
                                                double img_weight);

    void AddPnPMeasurementAnalyticNURBS(const Eigen::Vector3d &visual_map_point,
                                        const Eigen::Vector2d &pixel_obs,
                                        int64_t cur_img_timestamp,
                                        const SO3d &S_VtoI, const Eigen::Vector3d &p_VinI,
                                        const CameraGeometry &camera_geometry,
                                        double img_weight, double cost_scale = 1.0);

    void AddPhotometricMeasurementAutoDiffNURBS(const double &prev_pixel_intensity,
                                                const Eigen::Vector3d &visual_map_point,
                                                const cv::Mat &cur_img, int64_t cur_img_timestamp,
                                                const SO3d &S_VtoI, const Eigen::Vector3d &p_VinI, const Eigen::Matrix3d &K,
                                                double img_weight);

    void AddPhotometricMeasurementAnalyticNURBS(
        // const double &prev_pixel_intensity,
        int patch_size_half, int scale, int level,
        float *patch,
        const Eigen::Vector3d &visual_map_point,
        const cv::Mat &cur_img, int64_t cur_img_timestamp,
        const SO3d &S_VtoI, const Eigen::Vector3d &p_VinI, const Eigen::Matrix3d &K,
        double img_weight);

    void AddRalativeLoamFeatureAnalytic(const PointCorrespondence &pc,
                                        double weight,
                                        bool marg_this_factor = false);

    void AddLoamFeatureOptMapPoseAnalytic(const PointCorrespondence &pc,
                                          double *S_ImtoG, double *p_IminG,
                                          double weight,
                                          bool marg_this_factor = false);

    void AddImageFeatureAnalytic(const double ti, const Eigen::Vector3d &pi,
                                 const double tj, const Eigen::Vector3d &pj,
                                 double *inv_depth, bool fixed_depth = false,
                                 bool marg_this_fearure = false);

    void AddImage3D2DAnalytic(const double ti, const Eigen::Vector3d &pi,
                              double *p_G, bool fixed_p_G = false,
                              bool marg_this_fearure = false);

    void AddImageFeatureOnePoseAnalytic(
        const Eigen::Vector3d &p_i, const SO3d &S_IitoG,
        const Eigen::Vector3d &p_IiinG, const double t_j,
        const Eigen::Vector3d &p_j, double *inv_depth, bool fixed_depth = false,
        bool marg_this_fearure = false);

    void AddImageDepthAnalytic(const Eigen::Vector3d &p_i,
                               const Eigen::Vector3d &p_j, const SO3d &S_CitoCj,
                               const Eigen::Vector3d &p_CiinCj,
                               double *inv_depth);

    void AddEpipolarFactorAnalytic(const double t_i, const Eigen::Vector3d &x_i,
                                   const Eigen::Vector3d &x_k,
                                   const SO3d &S_GtoCk,
                                   const Eigen::Vector3d &p_CkinG, double weight,
                                   bool marg_this_fearure = false);

    void AddMarginalizationFactor(
        MarginalizationInfo::Ptr &last_marginalization_info,
        std::vector<double *> &last_marginalization_parameter_blocks);

    void AddPoseMeasurementAutoDiff(const PoseData &pose_data, double pos_weight,
                                    double rot_weight);

    void AddPoseMeasurementAutoDiffNURBS(const PoseData &pose_data, double pos_weight,
                                         double rot_weight);

    void AddPoseMeasurementAnalyticDiffNURBS(const PoseData &pose_data, double pos_weight,
                                             double rot_weight);

    void AddRelativePoseMeasurementAnalyticDiffNURBS(const PoseData &pose_data, double pos_weight,
                                                     double rot_weight);

    void Add6DofLocalVelocityAutoDiff(
        const double timestamp, const Eigen::Matrix<double, 6, 1> &local_gyro_vel,
        double gyro_weight, double velocity_weight);

    void Add6DofLocalVelocityAutoDiffNURBS(
        const double timestamp, const Eigen::Matrix<double, 6, 1> &local_gyro_vel,
        double gyro_weight, double velocity_weight);

    void AddCallback(const std::vector<std::string> &descriptions,
                     const std::vector<size_t> &block_size,
                     std::vector<double *> &param_block);

    ceres::Solver::Summary PreSolve(int max_iterations = 50, bool progress = false,
                                    int num_threads = -1);

    ceres::Solver::Summary Solve(int max_iterations = 50, bool progress = false,
                                 int num_threads = -1);

    ControlPointCovarianceResult ComputeControlPointCovariances(
        const std::vector<int64_t> &trajectory_times_ns);

    CausalObservabilityDiagnostics ComputeCausalObservabilityDiagnostics();

    CameraRobustRisk ComputeCameraRobustRisk() const;

    ProcessProjectionResult ComputeRobustProcessProjections(
        const std::vector<size_t> &support_starts,
        double translation_spectral_density,
        double rotation_spectral_density);

    void PrepareMarginalizationInfo(ResidualType r_type,
                                    ceres::CostFunction *cost_function,
                                    ceres::LossFunction *loss_function,
                                    std::vector<double *> &parameter_blocks,
                                    std::vector<int> &drop_set);

    void SaveMarginalizationInfo(MarginalizationInfo::Ptr &marg_info_out,
                                 std::vector<double *> &marg_param_blocks_out);

    const ResidualSummary &GetResidualSummary() const
    {
      return residual_summary_;
    }

    void AddControlPointsNURBS(size_t start_idx,
                               std::vector<double *> &vec, bool addPosKnot = false);

    void AddControlPointsNURBS(size_t start_idx1, size_t start_idx2,
                               std::vector<double *> &vec, bool addPosKnot = false);

    void AddRobustWnoaProcessFactors(
        size_t support_start, double translation_spectral_density,
        double rotation_spectral_density,
        const Eigen::Matrix<double, 6, 6> &translation_sqrt_weight,
        const Eigen::Matrix<double, 6, 6> &rotation_sqrt_weight,
        bool add_translation = true, bool add_rotation = true,
        double robust_cost_scale = 1.0, bool translation_wnoj = false,
        bool rotation_wnoj = false);

  private:
    void AddControlPoints(const SplineMeta<SplineOrder> &spline_meta,
                          std::vector<double *> &vec, bool addPosKnot = false);

    void PrepareMarginalizationInfo(ResidualType r_type,
                                    const SplineMeta<SplineOrder> &spline_meta,
                                    ceres::CostFunction *cost_function,
                                    ceres::LossFunction *loss_function,
                                    std::vector<double *> &parameter_blocks,
                                    std::vector<int> &drop_set_wo_ctrl_point);

    void PrepareMarginalizationInfo(ResidualType r_type,
                                    ceres::CostFunction *cost_function,
                                    ceres::LossFunction *loss_function,
                                    std::vector<double *> &parameter_blocks,
                                    std::vector<int> &drop_set_w_ctrl_point, bool is_lidar_inertial);

    bool IsParamUpdated(const double *values) const;

  public:
    TrajectoryEstimatorOptions options;

  private:
    Trajectory::Ptr trajectory_;

    std::shared_ptr<ceres::Problem> problem_;
    ceres::LocalParameterization *analytic_local_parameterization_;
    ceres::HomogeneousVectorParameterization *homo_vec_local_parameterization_;

    ceres::LocalParameterization *auto_diff_local_parameterization_;

    std::map<SensorType, double *> t_offset_ns_opt_params_;

    int fixed_control_point_index_;

    // Marginalization
    MarginalizationInfo::Ptr marginalization_info_;

    // for debug
    ResidualSummary residual_summary_;

    SensorObservability ComputeSensorObservability(
        const std::vector<ceres::ResidualBlockId> &residual_blocks,
        int residual_dimension, double robust_loss_scale) const;
    static ObservabilitySpectrum ComputeObservabilitySpectrum(
        const Eigen::MatrixXd &information);

    std::vector<ceres::ResidualBlockId> lidar_residual_blocks_;
    std::vector<ceres::ResidualBlockId> camera_residual_blocks_;

    bool callback_needs_state_;
    std::vector<std::unique_ptr<ceres::IterationCallback>> callbacks_;
  };

} // namespace cocolic
