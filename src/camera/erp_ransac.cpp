#include "erp_ransac.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <limits>

namespace cocolic {
namespace {
// Score epipolar planes without triangulation: depth is unobservable at rest
// or during pure rotation, but these frames must not lose their visual tracks.
class SphericalEpipolarProblem : public opengv::sac::SampleConsensusProblem<Eigen::Matrix3d> {
 public:
  SphericalEpipolarProblem(opengv::relative_pose::CentralRelativeAdapter& adapter,
                          bool random)
      : SampleConsensusProblem(random), adapter_(adapter) {
    setUniformIndices(adapter.getNumberCorrespondences());
  }
  int getSampleSize() const override { return 8; }
  bool computeModelCoefficients(const std::vector<int>& indices,
                                model_t& model) const override {
    model = opengv::relative_pose::eightpt(adapter_, indices);
    return model.allFinite() && model.norm() > 0.;
  }
  void optimizeModelCoefficients(const std::vector<int>& inliers,
      const model_t& initial, model_t& result) override {
    result = opengv::relative_pose::eightpt(adapter_, inliers);
  }
  void getSelectedDistancesToModel(const model_t& model,
      const std::vector<int>& indices, std::vector<double>& scores) const override {
    for (int i : indices) {
      const auto first = adapter_.getBearingVector1(i);
      const auto second = adapter_.getBearingVector2(i);
      const Eigen::Vector3d n1 = model * second;
      const Eigen::Vector3d n2 = model.transpose() * first;
      const double denominator = std::min(n1.norm(), n2.norm());
      scores.push_back(denominator > 0. ? std::abs(first.dot(n1)) / denominator
                                       : std::numeric_limits<double>::infinity());
    }
  }
 private:
  opengv::relative_pose::CentralRelativeAdapter& adapter_;
};

opengv::bearingVectors_t Bearings(const CameraGeometry& camera,
                                 const std::vector<cv::Point2f>& pixels) {
  opengv::bearingVectors_t bearings;
  bearings.reserve(pixels.size());
  for (const auto& pixel : pixels) {
    Eigen::Vector3d ray;
    if (!camera.unproject(Eigen::Vector2d(pixel.x, pixel.y), 1., ray))
      throw std::invalid_argument("Non-finite ERP correspondence");
    bearings.push_back(ray.normalized());
  }
  return bearings;
}

double AngularThreshold(const CameraGeometry& camera, double pixels) {
  // Equatorial pixel angle; the same angular tolerance applies over the sphere.
  const double angle = pixels * std::max(2. * M_PI / camera.width(),
                                         M_PI / camera.height());
  return 1. - std::cos(angle);
}
}

std::vector<unsigned char> ErpRelativeInliers(
    const CameraGeometry& camera, const std::vector<cv::Point2f>& previous,
    const std::vector<cv::Point2f>& current, bool deterministic) {
  if (previous.size() != current.size())
    throw std::invalid_argument("ERP relative correspondence size mismatch");
  std::vector<unsigned char> mask(current.size(), 0);
  auto first = Bearings(camera, previous);
  auto second = Bearings(camera, current);
  opengv::relative_pose::CentralRelativeAdapter adapter(first, second);
  using Problem = SphericalEpipolarProblem;
  auto problem = std::make_shared<Problem>(adapter, !deterministic);
  if (current.size() < static_cast<size_t>(problem->getSampleSize()))
    return mask;
  opengv::sac::Ransac<Problem> ransac;
  ransac.sac_model_ = problem;
  ransac.threshold_ = std::sin(std::max(2. * M_PI / camera.width(), M_PI / camera.height()));
  ransac.max_iterations_ = 200;
  ransac.probability_ = .997;
  if (ransac.computeModel())
    for (int i : ransac.inliers_) mask[i] = 1;
  return mask;
}

std::vector<unsigned char> ErpAbsoluteInliers(
    const CameraGeometry& camera, const std::vector<Eigen::Vector3d>& world,
    const std::vector<cv::Point2f>& pixels, bool deterministic) {
  if (world.size() != pixels.size())
    throw std::invalid_argument("ERP absolute correspondence size mismatch");
  std::vector<unsigned char> mask(pixels.size(), 0);
  auto bearings = Bearings(camera, pixels);
  opengv::points_t points(world.begin(), world.end());
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearings, points);
  using Problem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
  auto problem = std::make_shared<Problem>(adapter, Problem::KNEIP, !deterministic);
  if (pixels.size() < static_cast<size_t>(problem->getSampleSize()))
    return mask;
  opengv::sac::Ransac<Problem> ransac;
  ransac.sac_model_ = problem;
  ransac.threshold_ = AngularThreshold(camera, 1.5);
  ransac.max_iterations_ = 200;
  ransac.probability_ = .99;
  if (ransac.computeModel())
    for (int i : ransac.inliers_) mask[i] = 1;
  return mask;
}
}
