#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <simple_nnet/Layer.h>

#define TEST_BATCH_SIZE 10

using Eigen::MatrixXf;

class MockLayer : public ILayer {
public:
  MockLayer(size_t inputs, size_t outputs, size_t batchSize) {
    _inputs = inputs;
    _outputs = outputs;
    _batchSize = batchSize;
    _activations = MatrixXf::Constant(_batchSize, _outputs, 1.0);
    _weights = MatrixXf::Constant(_inputs, outputs, 1.0);
  }

  virtual const MatrixXf activationUpdate(const MatrixXf &input) {
    return _activations;
  };
  virtual MatrixXf activation(const MatrixXf &input) const {
    return MatrixXf::Constant(_batchSize, _outputs, 1.0);
  }
  virtual MatrixXf dActivation() {
    return MatrixXf::Constant(_batchSize, _outputs, 1.0);
  }
  virtual size_t inputWidth() const { return _inputs; }
  virtual size_t outputWidth() const { return _outputs; }
  virtual const MatrixXf &weights() const { return _weights; }
  virtual void propagate(MatrixXf &error) {}
  virtual void propagate(ILayer &previous) {}
  virtual void applyGradient(float eps) {}
  virtual void print() const {}
  virtual const MatrixXf &delta() const { return _activations; }

  size_t _inputs;
  size_t _outputs;
  size_t _batchSize;
  MatrixXf _activations;
  MatrixXf _weights;
};

TEST_CASE("Single layer tests", "[main]") {
  
  SigmoidLayer l1(0, MatrixXf::Constant(2, 3, 1.0f),
               RowVectorXf::Constant(3, 1.0f));
  REQUIRE(l1.inputWidth() == 2);
  REQUIRE(l1.outputWidth() == 3);

  MatrixXf in = MatrixXf::Constant(TEST_BATCH_SIZE, 2, 1.0f);
  MatrixXf expectedOut = MatrixXf::Constant(TEST_BATCH_SIZE, 3, sigmoid(3.0));
  MatrixXf out = l1.activationUpdate(in);
  REQUIRE(out == expectedOut);

  MatrixXf expectedDAct = MatrixXf::Constant(TEST_BATCH_SIZE, 3, dSigmoid(3.0f));
  REQUIRE(l1.dActivation() == expectedDAct);

  MatrixXf errors = MatrixXf::Constant(TEST_BATCH_SIZE, 3, 1.0f);

  MatrixXf expectedDelta = MatrixXf::Constant(TEST_BATCH_SIZE, 3, dSigmoid(3.0f));
  l1.propagate(errors);
  REQUIRE(l1.delta() == expectedDelta);

  MockLayer mockLayer(3, 2, TEST_BATCH_SIZE);

  l1.propagate(mockLayer);

  l1.applyGradient(1.0);
  MatrixXf expectedWeights = MatrixXf::Constant(2, 3, 1) - in.transpose() * expectedDelta;
  REQUIRE(l1.weights().isApprox(expectedWeights));
}