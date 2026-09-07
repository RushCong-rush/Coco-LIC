#include "camera/erp_ransac.h"
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <stdexcept>

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR, 1024, 512);
  std::mt19937 generator(7);
  std::normal_distribution<double> normal;
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.12, Eigen::Vector3d(.2, 1., .1).normalized()).toRotationMatrix();
  for (int scenario = 0; scenario < 6; ++scenario) {
    const Eigen::Matrix3d r = scenario % 4 == 0 ? Eigen::Matrix3d::Identity() : rotation;
    const Eigen::Vector3d t = (scenario == 2 || scenario == 3) ? Eigen::Vector3d(.3, .1, .2) : Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> world;
    std::vector<cv::Point2f> first, second;
    for (int i = 0; i < 240; ++i) {
      Eigen::Vector3d ray(normal(generator), normal(generator), normal(generator));
      // Include both sides of the seam and high latitudes, not just z-positive rays.
      if (i < 10) ray = Eigen::Vector3d((i % 2 ? 1. : -1.) * .0001, .1, -1.);
      if (i >= 10 && i < 20) ray = Eigen::Vector3d(.03, i % 2 ? 1. : -1., .02);
      Eigen::Vector3d p = ray.normalized() * (3. + .04 * i);
      world.push_back(p);
      Eigen::Vector2d a, b;
      Require(camera.project(p, a) && camera.project(r.transpose() * (p-t), b), "projection");
      if (scenario >= 3) b += Eigen::Vector2d(.15*normal(generator), .15*normal(generator));
      first.emplace_back(a.x(), a.y());
      second.emplace_back(b.x(), b.y());
    }
    for (int i = 180; i < 240; ++i) second[i] = second[i-130];
    const auto relative = cocolic::ErpRelativeInliers(camera, first, second, true);
    const auto absolute = cocolic::ErpAbsoluteInliers(camera, world, second, true);
    Require(relative == cocolic::ErpRelativeInliers(camera, first, second, true), "relative repeatability");
    Require(absolute == cocolic::ErpAbsoluteInliers(camera, world, second, true), "absolute repeatability");
    for (const auto* mask : {&relative, &absolute}) {
      int kept = 0, false_kept = 0;
      for (int i = 0; i < 240; ++i)
        if ((*mask)[i]) (i < 180 ? kept : false_kept)++;
      std::cout << "scenario=" << scenario << (mask == &relative ? " relative" : " absolute")
                << " inliers=" << kept << "/180 false=" << false_kept << "/60" << std::endl;
      Require(kept >= 165, "Too many valid spherical observations rejected");
      Require(false_kept <= 6, "Too many outliers retained");
    }
  }
  Require(cocolic::ErpRelativeInliers(camera, {}, {}, true).empty(), "empty relative");
  Require(cocolic::ErpAbsoluteInliers(camera, {}, {}, true).empty(), "empty absolute");
  const std::vector<cv::Point2f> few(3, cv::Point2f(1, 1));
  auto mask = cocolic::ErpAbsoluteInliers(camera, std::vector<Eigen::Vector3d>(3, Eigen::Vector3d::Ones()), few, true);
  Require(std::count(mask.begin(), mask.end(), 1) == 0, "insufficient P3P sample");
  std::cout << "ERP RANSAC tests passed" << std::endl;
}
