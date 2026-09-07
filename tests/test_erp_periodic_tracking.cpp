#include "camera/rgb_map/rgbmap_tracker.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace {
bool Check(cocolic::CameraModel model, int shift) {
  constexpr int width = 1024, height = 512;
  const bool erp = model == cocolic::CameraModel::EQUIRECTANGULAR;
  cv::Mat previous(height, width, CV_8UC1);
  cv::RNG rng(0);
  rng.fill(previous, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(previous, previous, cv::Size(5, 5), .8);
  cv::Mat current(height, width, CV_8UC1);
  for (int x = 0; x < width; ++x)
    previous.col(x).copyTo(current.col((x + shift + width) % width));

  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  k(0, 0) = k(1, 1) = 320.;
  k(0, 2) = width / 2.;
  k(1, 2) = height / 2.;
  auto camera = std::make_shared<cocolic::CameraGeometry>(model, width, height, k);
  auto image = std::make_shared<Image_frame>(k);
  image->set_camera_geometry(camera);
  image->m_img_cols = width;
  image->m_img_rows = height;
  image->m_timestamp = .1;
  image->m_img_gray = current;
  cv::cvtColor(current, image->m_img, cv::COLOR_GRAY2BGR);
  image->set_pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

  Rgbmap_tracker tracker(true);
  tracker.last_img = previous;
  tracker.m_last_frame_time = 0.;
  std::vector<std::shared_ptr<RGB_pts>> points;
  std::vector<cv::Point2f> pixels;
  std::vector<bool> crossing;
  const std::vector<int> xs = erp
      ? std::vector<int>{3, 8, 100, 250, 450, 700, 950, 1015, 1020}
      : std::vector<int>{100, 250, 450, 700, 900};
  for (int x : xs) {
    for (int y = 60; y < height - 60; y += 32) {
      auto point = std::make_shared<RGB_pts>();
      point->m_pt_index = points.size();
      Eigen::Vector3d world;
      camera->unproject(Eigen::Vector2d(x, y), 5., world);
      point->set_pos(world);
      tracker.m_map_rgb_pts_in_last_frame_pos[point.get()] = cv::Point2f(x, y);
      points.push_back(point);
      pixels.emplace_back(x, y);
      crossing.push_back(x + shift < 0 || x + shift >= width);
    }
  }
  tracker.update_last_tracking_vector_and_ids();
  tracker.track_img(image, -1., 1);

  int counts[2] = {}, correct[2] = {};
  for (size_t i = 0; i < points.size(); ++i) {
    const int group = crossing[i] ? 1 : 0;
    ++counts[group];
    const auto found = tracker.m_map_rgb_pts_in_current_frame_pos.find(points[i].get());
    if (found == tracker.m_map_rgb_pts_in_current_frame_pos.end()) continue;
    const Eigen::Vector2d truth(camera->wrapPixelU(pixels[i].x + shift), pixels[i].y);
    const Eigen::Vector2d estimate(found->second.x, found->second.y);
    if (estimate.x() >= 0 && estimate.x() < width &&
        camera->pixelDifference(estimate, truth).norm() < 1.) ++correct[group];
  }
  for (int group = 0; group < 2; ++group)
    std::cout << (erp ? "ERP" : "pinhole") << " shift=" << shift
              << (group ? " crossing" : " interior") << " total=" << counts[group]
              << " correct_within_1px=" << correct[group] << '\n';
  return correct[0] == counts[0] && correct[1] == counts[1];
}
}

int main() {
  cv::setNumThreads(1);
  cv::setRNGSeed(0);
  bool passed = true;
  for (auto model : {cocolic::CameraModel::EQUIRECTANGULAR, cocolic::CameraModel::PINHOLE})
    for (int shift : {12, -12})
      passed = Check(model, shift) && passed;
  return passed ? 0 : 1;
}
