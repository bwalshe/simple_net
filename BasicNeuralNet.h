#ifndef BASIC_NEURAL_NET_H
#define BASIC_NEURAL_NET_H

#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "relu.h"

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class Network {

public:
  Network(const int inputWidth, const int outputWidth) {
    weights.push_back(MatrixXf::Random(inputWidth, outputWidth));
    biases.push_back(RowVectorXf::Random(outputWidth));
  }

  int layerSize(const int i) { return weights[i].cols(); }

  void addLayer(const int width) {
    int oldOutputWidth = layerSize(numLayers() - 1);
    weights.push_back(MatrixXf::Random(oldOutputWidth, width));
    biases.push_back(RowVectorXf::Random(width));
  }

  size_t numLayers() { return weights.size(); }

  void printStructure() {
    for (auto i = 0; i < numLayers(); ++i) {
      std::cout << "=== Layer " << i << " ======" << std::endl;
      std::cout << "weights: " << std::endl << weights[i] << std::endl;
      std::cout << "bias: " << biases[i] << std::endl;
      std::cout << "====================" << std::endl << std::endl;
    }
  }

  float backprop(const MatrixXf &x, const MatrixXf &y, const float eps) {
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
        activations[numLayers() - 1].colwise().sum().transpose() * delta;
    for (auto i = numLayers() - 1; i > 0; --i) {
      RowVectorXf zsum = zeds[i - 1].unaryExpr(std::ref(dRelu)).colwise().sum() / batchSize;
      RowVectorXf dpart = delta * weights[i].transpose();
      delta = dpart.cwiseProduct(zsum);
      biases[i - 1] += delta;
      RowVectorXf activationSum =
          activations[i - 1].colwise().sum() / batchSize;
      weights[i - 1] += activationSum.transpose() * delta;
    }
    return (activations.back() - y).squaredNorm() / batchSize;
  }

  MatrixXf feed(const MatrixXf &x) {
    MatrixXf out(x);
    for (auto i = 0; i < numLayers(); ++i) {
      out = (out * weights[i]).rowwise() + biases[i];
    }
    return out;
  }

private:
  std::vector<MatrixXf> weights;
  std::vector<RowVectorXf> biases;

  int inputWidth;

  static MatrixXf activation(const MatrixXf &m) {
    return m.unaryExpr(std::ref(relu));
  }

  static MatrixXf dActivation(MatrixXf &m) {
    return m.unaryExpr(std::ref(dRelu));
  }
};

#endif //BASIC_NEURAL_NET_H