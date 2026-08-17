/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cocolic
{

enum class CameraModel
{
  PINHOLE,
  EQUIRECTANGULAR,
};

inline CameraModel CameraModelFromString(const std::string &name)
{
  if (name == "pinhole")
    return CameraModel::PINHOLE;
  if (name == "equirectangular" || name == "erp")
    return CameraModel::EQUIRECTANGULAR;
  throw std::invalid_argument("Unsupported camera model: " + name);
}

class CameraGeometry
{
public:
  CameraGeometry(CameraModel model, int width, int height,
                 const Eigen::Matrix3d &intrinsics = Eigen::Matrix3d::Identity())
      : model_(model), width_(width), height_(height), intrinsics_(intrinsics)
  {
    if (width_ <= 1 || height_ <= 1)
      throw std::invalid_argument("Camera image dimensions must be greater than one");
  }

  CameraModel model() const { return model_; }
  int width() const { return width_; }
  int height() const { return height_; }
  bool isEquirectangular() const { return model_ == CameraModel::EQUIRECTANGULAR; }

  bool project(const Eigen::Vector3d &point_camera, Eigen::Vector2d &pixel,
               double *depth = nullptr) const
  {
    if (!point_camera.allFinite())
      return false;

    if (model_ == CameraModel::PINHOLE)
    {
      const double z = point_camera.z();
      if (z <= kEpsilon)
        return false;
      pixel.x() = intrinsics_(0, 0) * point_camera.x() / z + intrinsics_(0, 2);
      pixel.y() = intrinsics_(1, 1) * point_camera.y() / z + intrinsics_(1, 2);
      if (depth)
        *depth = z;
      return pixel.allFinite();
    }

    const double x = point_camera.x();
    const double y = point_camera.y();
    const double z = point_camera.z();
    const double radius = point_camera.norm();
    const double horizontal_radius = std::hypot(x, z);
    if (radius <= kEpsilon || horizontal_radius <= kPoleEpsilon)
      return false;

    const double longitude = std::atan2(x, z);
    const double latitude = std::atan2(-y, horizontal_radius);
    pixel.x() = wrapPixelU(width_ * (longitude + kPi) / (2.0 * kPi));
    pixel.y() = height_ * (0.5 - latitude / kPi);
    if (depth)
      *depth = radius;
    return pixel.allFinite();
  }

  bool unproject(const Eigen::Vector2d &pixel, double depth,
                 Eigen::Vector3d &point_camera) const
  {
    if (!pixel.allFinite() || !std::isfinite(depth) || depth <= kEpsilon)
      return false;

    if (model_ == CameraModel::PINHOLE)
    {
      const double fx = intrinsics_(0, 0);
      const double fy = intrinsics_(1, 1);
      if (std::abs(fx) <= kEpsilon || std::abs(fy) <= kEpsilon)
        return false;
      point_camera = Eigen::Vector3d(
          (pixel.x() - intrinsics_(0, 2)) * depth / fx,
          (pixel.y() - intrinsics_(1, 2)) * depth / fy, depth);
      return point_camera.allFinite();
    }

    const double longitude = 2.0 * kPi * wrapPixelU(pixel.x()) / width_ - kPi;
    const double latitude = kPi * (0.5 - pixel.y() / height_);
    const double cos_latitude = std::cos(latitude);
    point_camera = depth * Eigen::Vector3d(
        cos_latitude * std::sin(longitude), -std::sin(latitude),
        cos_latitude * std::cos(longitude));
    return point_camera.allFinite();
  }

  bool projectJacobian(const Eigen::Vector3d &point_camera,
                       Eigen::Matrix<double, 2, 3> &jacobian) const
  {
    if (!point_camera.allFinite())
      return false;

    const double x = point_camera.x();
    const double y = point_camera.y();
    const double z = point_camera.z();
    if (model_ == CameraModel::PINHOLE)
    {
      if (z <= kEpsilon)
        return false;
      const double z2 = z * z;
      jacobian << intrinsics_(0, 0) / z, 0.0, -intrinsics_(0, 0) * x / z2,
          0.0, intrinsics_(1, 1) / z, -intrinsics_(1, 1) * y / z2;
      return jacobian.allFinite();
    }

    const double horizontal_radius2 = x * x + z * z;
    const double horizontal_radius = std::sqrt(horizontal_radius2);
    const double radius2 = horizontal_radius2 + y * y;
    if (radius2 <= kEpsilon * kEpsilon || horizontal_radius <= kPoleEpsilon)
      return false;

    jacobian << width_ * z / (2.0 * kPi * horizontal_radius2), 0.0,
        -width_ * x / (2.0 * kPi * horizontal_radius2),
        -height_ * x * y / (kPi * radius2 * horizontal_radius),
        height_ * horizontal_radius / (kPi * radius2),
        -height_ * y * z / (kPi * radius2 * horizontal_radius);
    return jacobian.allFinite();
  }

  double wrapPixelU(double u) const
  {
    if (!isEquirectangular())
      return u;
    u = std::fmod(u, static_cast<double>(width_));
    return u < 0.0 ? u + width_ : u;
  }

  Eigen::Vector2d pixelDifference(const Eigen::Vector2d &observed,
                                  const Eigen::Vector2d &predicted) const
  {
    Eigen::Vector2d difference = observed - predicted;
    if (isEquirectangular())
    {
      const double half_width = 0.5 * width_;
      difference.x() = std::fmod(difference.x() + half_width,
                                 static_cast<double>(width_));
      if (difference.x() < 0.0)
        difference.x() += width_;
      difference.x() -= half_width;
    }
    return difference;
  }

private:
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kEpsilon = 1e-12;
  static constexpr double kPoleEpsilon = 1e-8;

  CameraModel model_;
  int width_;
  int height_;
  Eigen::Matrix3d intrinsics_;
};

} // namespace cocolic
