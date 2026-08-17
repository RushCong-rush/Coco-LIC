#include <camera/camera_geometry.h>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

void Require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

Eigen::Vector2d Residual(const cocolic::CameraGeometry &camera,
                         const Eigen::Vector3d &point,
                         const Eigen::Vector2d &observation)
{
  Eigen::Vector2d prediction;
  Require(camera.project(point, prediction), "ERP projection failed");
  return camera.pixelDifference(observation, prediction);
}

void CheckResidualJacobian(const Eigen::Vector3d &point,
                           const Eigen::Vector2d &observation)
{
  const cocolic::CameraGeometry camera(cocolic::CameraModel::EQUIRECTANGULAR,
                                       1024, 512);
  Eigen::Matrix<double, 2, 3> analytic;
  Require(camera.projectJacobian(point, analytic), "ERP Jacobian failed");
  analytic = -analytic;

  Eigen::Matrix<double, 2, 3> numeric;
  constexpr double step = 1e-6;
  for (int axis = 0; axis < 3; ++axis)
  {
    Eigen::Vector3d plus = point;
    Eigen::Vector3d minus = point;
    plus(axis) += step;
    minus(axis) -= step;
    numeric.col(axis) =
        (Residual(camera, plus, observation) -
         Residual(camera, minus, observation)) /
        (2.0 * step);
  }
  Require((analytic - numeric).cwiseAbs().maxCoeff() < 2e-6,
          "ERP visual residual Jacobian mismatch");
}

} // namespace

int main()
{
  try
  {
    CheckResidualJacobian(Eigen::Vector3d(1.3, -0.4, 3.7),
                          Eigen::Vector2d(570.0, 244.0));
    CheckResidualJacobian(Eigen::Vector3d(0.01, 0.2, -4.0),
                          Eigen::Vector2d(1.0, 248.0));
  }
  catch (const std::exception &error)
  {
    std::cerr << "erp_visual_residual_test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "erp_visual_residual_test passed\n";
  return 0;
}
