#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <simple_nnet/BasicNeuralNet.h>

Network::Network(const int inputWidth, const int outputWidth) {
  _layers.push_back(std::make_unique<ReluLayer>(0, inputWidth, outputWidth));
}

void Network::addLayer(const int width) {
  int oldOutputWidth = layerSize(numLayers() - 1);
  _layers.push_back(
      std::make_unique<ReluLayer>(_layers.size(), oldOutputWidth, width));
}

void Network::printStructure() const {
  for (auto layer = _layers.begin(); layer != _layers.end(); ++layer) {
    layer->get()->print();
  }
}

float Network::backprop(const MatrixXf &x, const MatrixXf &y, const float eps) {
  auto batchSize = x.rows();
  MatrixXf out(x);
  for (auto layer = _layers.begin(); layer != _layers.end(); ++layer) {
    out = layer->get()->activationUpdate(out);
  }
  MatrixXf errors = out - y;
  _layers.back()->propagate(errors);

  for (auto i = numLayers() - 1; i > 0; --i) {
    _layers[i - 1]->propagate(*_layers[i]);
  }
  for (auto i = 0; i < numLayers(); ++i) {
    _layers[i]->applyGradient(eps);
  }
  return errors.squaredNorm() / batchSize;
}

MatrixXf Network::feed(const MatrixXf &x) const {
  MatrixXf out(x);
  for (auto layer = _layers.begin(); layer != _layers.end(); ++layer) {
    out = layer->get()->activation(out);
  }
  std::cout << std::endl;
  return out;
}