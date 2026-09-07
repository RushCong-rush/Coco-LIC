#include "camera/rgb_map/rgbmap_tracker.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void Check(cocolic::CameraModel model, double predicted_u, double observed_u,
           bool expected_keep) {
  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  k(0, 0) = k(1, 1) = 320.;
  k(0, 2) = 512.;
  k(1, 2) = 256.;
  auto geometry = std::make_shared<cocolic::CameraGeometry>(model, 1024, 512, k);
  auto image = std::make_shared<Image_frame>(k);
  image->set_camera_geometry(geometry);
  image->m_img_cols = 1024;
  image->m_img_rows = 512;
  image->set_pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  RGB_pts point;
  point.m_pt_index = 0;
  Eigen::Vector3d world;
  if (!geometry->unproject(Eigen::Vector2d(predicted_u, 256.), 5., world))
    throw std::runtime_error("Invalid test projection");
  point.set_pos(world);
  Rgbmap_tracker tracker(true);
  tracker.m_map_rgb_pts_in_last_frame_pos[&point] = cv::Point2f(observed_u, 256.);
  Global_map empty_map(0, true);
  tracker.update_and_append_track_pts(image, empty_map, 20., 1000000);
  const bool kept = tracker.m_map_rgb_pts_in_last_frame_pos.count(&point) != 0;
  std::cout << (geometry->isEquirectangular() ? "ERP" : "pinhole")
            << " predicted=" << predicted_u << " observed=" << observed_u
            << " retained=" << kept << " expected=" << expected_keep << '\n';
  if (kept != expected_keep)
    throw std::runtime_error("Tracking append gate mismatch");
}
}

int main() {
  try {
    Check(cocolic::CameraModel::EQUIRECTANGULAR, 1., 1023., true);
    Check(cocolic::CameraModel::EQUIRECTANGULAR, 1023., 1., true);
    Check(cocolic::CameraModel::EQUIRECTANGULAR, 500., 502., true);
    Check(cocolic::CameraModel::EQUIRECTANGULAR, 500., 520., false);
    Check(cocolic::CameraModel::PINHOLE, 500., 502., true);
    Check(cocolic::CameraModel::PINHOLE, 500., 520., false);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
