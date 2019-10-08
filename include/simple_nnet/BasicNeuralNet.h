#ifndef BASIC_NEURAL_NET_H
#define BASIC_NEURAL_NET_H

#include <memory>
#include <vector>
#include <simple_nnet/Layer.h>

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class Network {

public:
  Network(const int inputWidth, const int outputWidth);

  int layerSize(const int i) const { return _layers[i]->outputWidth(); }

  void addLayer(const int width);

  size_t numLayers() const { return _layers.size(); }

  void printStructure() const;

  float backprop(const MatrixXf &x, const MatrixXf &y, const float eps);

  MatrixXf feed(const MatrixXf &x) const;

private:
  std::vector<std::unique_ptr<ILayer>> _layers;
  int inputWidth;
};

#endif // BASIC_NEURAL_NET_H