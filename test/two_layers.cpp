#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <simple_nnet/Layer.h>

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

///
/// This really just checkes that the layers are working together
/// The computed numbers are a bit arbitrary.
///
TEST_CASE("Two layer tests", "[main]") {
  Eigen::MatrixXf w = MatrixXf::Constant(2, 2, 1);
  Eigen::RowVectorXf b = RowVectorXf::Constant(2, 1.0f);

  SigmoidLayer l1(0, w, b);
  SigmoidLayer l2(1, w, b);

  Eigen::MatrixXf in = MatrixXf::Constant(3, 2, 1.0f);
  Eigen::MatrixXf out1 = l1.activationUpdate(in);
  Eigen::MatrixXf out2 = l2.activationUpdate(out1);

  Eigen::MatrixXf errors = MatrixXf::Constant(3, 2, 1.0f);

  l2.propagate(errors);
  l1.propagate(l2);

  l2.applyGradient(0.1);
  l1.applyGradient(0.1);

  Eigen::MatrixXf expectedWeights1 = MatrixXf::Constant(2, 2, 0.99933f);

  REQUIRE(l1.weights().isApprox(expectedWeights1));
}