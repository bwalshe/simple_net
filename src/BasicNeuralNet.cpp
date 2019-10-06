#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <simple_nnet/BasicNeuralNet.h>
#include <simple_nnet/relu.h>

Network::Network(const int inputWidth, const int outputWidth) {
  weights.push_back(MatrixXf::Random(inputWidth, outputWidth));
  biases.push_back(RowVectorXf::Random(outputWidth));
}

void Network::addLayer(const int width) {
  int oldOutputWidth = layerSize(numLayers() - 1);
  weights.push_back(MatrixXf::Random(oldOutputWidth, width));
  biases.push_back(RowVectorXf::Random(width));
}

void Network::printStructure() const {
  for (auto i = 0; i < numLayers(); ++i) {
    std::cout << "=== Layer " << i << " ======" << std::endl;
    std::cout << "weights: " << std::endl << weights[i] << std::endl;
    std::cout << "bias: " << biases[i] << std::endl;
    std::cout << "====================" << std::endl << std::endl;
  }
}

float Network::backprop(const MatrixXf &x, const MatrixXf &y, const float eps) {
  auto batchSize = x.rows();
  std::vector<MatrixXf> zeds;
  std::vector<MatrixXf> activations;
  activations.push_back(x);
  for (auto i = 0; i < numLayers(); ++i) {
    auto z = (activations.back() * weights[i]).rowwise() + biases[i];
    zeds.push_back(z);
    activations.push_back(activation(z));
  }
  MatrixXf errors = activations.back() - y;
  MatrixXf dOutputActivation = dActivation(zeds.back());
  MatrixXf batchDeltas = errors.cwiseProduct(dOutputActivation);
  RowVectorXf delta = -0.1 * eps * batchDeltas.colwise().sum() / batchSize;
  biases.back() += delta;
  weights.back() +=
      activations[numLayers() - 1].colwise().sum().transpose() * delta / batchSize;
  for (auto i = numLayers() - 1; i > 0; --i) {
    RowVectorXf zsum = dActivation(zeds[i - 1]).colwise().sum() / batchSize;
    RowVectorXf dpart = delta * weights[i].transpose();
    delta = dpart.cwiseProduct(zsum);
    biases[i - 1] += delta;
    RowVectorXf activationSum = activations[i - 1].colwise().sum() / batchSize;
    weights[i - 1] += activationSum.transpose() * delta;
  }
  errors = activations.back() - y;
  return errors.squaredNorm() / batchSize;
}

MatrixXf Network::feed(const MatrixXf &x) const {
  MatrixXf out(x);
  for (auto i = 0; i < numLayers(); ++i) {
    out = activation((out * weights[i]).rowwise() + biases[i]);
  }
  return out;
}

MatrixXf Network::activation(const MatrixXf &m) {
  return m.unaryExpr(std::ref(relu));
}

MatrixXf Network::dActivation(MatrixXf &m) {
  return m.unaryExpr(std::ref(dRelu));
}