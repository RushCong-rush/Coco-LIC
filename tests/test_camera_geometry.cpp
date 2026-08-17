#include <camera/camera_geometry.h>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace
{

void Require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

void RequireNear(double actual, double expected, double tolerance, const char *message)
{
  if (std::abs(actual - expected) > tolerance)
  {
    std::cerr << message << ": actual=" << actual << ", expected=" << expected << '\n';
    throw std::runtime_error(message);
  }
}

void TestAxisDirections()
{
  const cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR,
                                       1024, 512);
  Eigen::Vector2d pixel;
  double range = 0.0;
  Require(camera.project(Eigen::Vector3d(0.0, 0.0, 2.0), pixel, &range),
          "+Z projection failed");
  RequireNear(pixel.x(), 512.0, 1e-12, "+Z longitude");
  RequireNear(pixel.y(), 256.0, 1e-12, "+Z latitude");
  RequireNear(range, 2.0, 1e-12, "ERP radial depth");

  Require(camera.project(Eigen::Vector3d(1.0, 0.0, 0.0), pixel),
          "+X projection failed");
  RequireNear(pixel.x(), 768.0, 1e-12, "+X longitude");
  Require(camera.project(Eigen::Vector3d(-1.0, 0.0, 0.0), pixel),
          "-X projection failed");
  RequireNear(pixel.x(), 256.0, 1e-12, "-X longitude");
  Require(camera.project(Eigen::Vector3d(0.0, 0.0, -1.0), pixel),
          "-Z projection failed");
  RequireNear(pixel.x(), 0.0, 1e-12, "-Z seam");

  Require(!camera.project(Eigen::Vector3d(0.0, 1.0, 0.0), pixel),
          "Exact pole must be rejected");
}

void TestRoundTrip()
{
  const cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR,
                                       1024, 512);
  std::mt19937 generator(7);
  std::uniform_real_distribution<double> distribution(-10.0, 10.0);
  for (int sample = 0; sample < 1000; ++sample)
  {
    Eigen::Vector3d point(distribution(generator), distribution(generator),
                          distribution(generator));
    if (point.norm() < 0.1 || std::hypot(point.x(), point.z()) < 0.1)
    {
      --sample;
      continue;
    }
    Eigen::Vector2d pixel;
    double range = 0.0;
    Require(camera.project(point, pixel, &range), "Random projection failed");
    Eigen::Vector3d reconstructed;
    Require(camera.unproject(pixel, range, reconstructed), "Random unprojection failed");
    Require((point - reconstructed).norm() < 1e-10, "ERP round trip mismatch");
  }
}

void TestJacobian()
{
  const cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR,
                                       1024, 512);
  const Eigen::Vector3d point(2.1, -0.7, 4.3);
  Eigen::Matrix<double, 2, 3> analytic;
  Require(camera.projectJacobian(point, analytic), "Analytic Jacobian failed");

  Eigen::Matrix<double, 2, 3> numeric;
  constexpr double step = 1e-6;
  for (int axis = 0; axis < 3; ++axis)
  {
    Eigen::Vector3d plus = point;
    Eigen::Vector3d minus = point;
    plus(axis) += step;
    minus(axis) -= step;
    Eigen::Vector2d plus_pixel;
    Eigen::Vector2d minus_pixel;
    Require(camera.project(plus, plus_pixel), "Positive finite difference failed");
    Require(camera.project(minus, minus_pixel), "Negative finite difference failed");
    numeric.col(axis) = camera.pixelDifference(plus_pixel, minus_pixel) / (2.0 * step);
  }
  Require((analytic - numeric).cwiseAbs().maxCoeff() < 2e-6,
          "ERP analytic Jacobian mismatch");
}

void TestSeamDistance()
{
  const cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR,
                                       1024, 512);
  const Eigen::Vector2d difference =
      camera.pixelDifference(Eigen::Vector2d(1.0, 100.0),
                             Eigen::Vector2d(1023.0, 98.0));
  RequireNear(difference.x(), 2.0, 1e-12, "Wrapped seam difference");
  RequireNear(difference.y(), 2.0, 1e-12, "Vertical pixel difference");
}

} // namespace

int main()
{
  try
  {
    TestAxisDirections();
    TestRoundTrip();
    TestJacobian();
    TestSeamDistance();
  }
  catch (const std::exception &error)
  {
    std::cerr << "camera_geometry_test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "camera_geometry_test passed\n";
  return 0;
}
