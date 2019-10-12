#pragma once

#include <memory>
#include <simple_nnet/Layer.h>
#include <vector>

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class Network {

  friend class NetworkBuilder;

public:
  class NetworkBuilder {
    friend class Network;

  public:
    template <class T> NetworkBuilder *addLayer(size_t width) {
      _layers.push_back(
          std::make_unique<T>(_layers.size(), _currentOutputWidth, width));
      _currentOutputWidth = width;
      return this;
    }
    Network build();

  private:
    NetworkBuilder(size_t inputWidth) : _currentOutputWidth(inputWidth) {}
    std::vector<std::unique_ptr<ILayer>> _layers;
    size_t _currentOutputWidth;
  };

  int layerSize(const int i) const { return _layers[i]->outputWidth(); }

  size_t numLayers() const { return _layers.size(); }

  void printStructure() const;

  float backprop(const MatrixXf &x, const MatrixXf &y, const float eps);

  MatrixXf feed(const MatrixXf &x) const;

  static NetworkBuilder builder(size_t inputWidth);

private:
  Network(std::vector<std::unique_ptr<ILayer>> &layers) {
    for (auto layer = layers.begin(); layer != layers.end(); ++layer) {
      _layers.push_back(std::move(*layer));
    }
  }

  std::vector<std::unique_ptr<ILayer>> _layers;
  int inputWidth;
};