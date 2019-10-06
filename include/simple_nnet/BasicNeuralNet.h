#ifndef BASIC_NEURAL_NET_H
#define BASIC_NEURAL_NET_H

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class Network {

public:
  Network(const int inputWidth, const int outputWidth);

  int layerSize(const int i) const { return weights[i].cols(); }

  void addLayer(const int width);

  size_t numLayers() const { return weights.size(); }

  void printStructure() const;

  float backprop(const MatrixXf &x, const MatrixXf &y, const float eps);

  MatrixXf feed(const MatrixXf &x) const;

private:
  std::vector<MatrixXf> weights;
  std::vector<RowVectorXf> biases;

  int inputWidth;

  static inline MatrixXf activation(const MatrixXf &m);

  static inline MatrixXf dActivation(MatrixXf &m);
};

#endif // BASIC_NEURAL_NET_H