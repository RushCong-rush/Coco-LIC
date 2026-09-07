#pragma once

#include "camera_geometry.h"
#include <opencv2/core/types.hpp>
#include <vector>

namespace cocolic {
// Masks refer to input order. No estimated pose is injected into the trajectory.
std::vector<unsigned char> ErpRelativeInliers(
    const CameraGeometry& camera, const std::vector<cv::Point2f>& previous,
    const std::vector<cv::Point2f>& current, bool deterministic);
std::vector<unsigned char> ErpAbsoluteInliers(
    const CameraGeometry& camera, const std::vector<Eigen::Vector3d>& world,
    const std::vector<cv::Point2f>& pixels, bool deterministic);
}
