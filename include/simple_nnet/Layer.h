#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <iostream>
#include <math.h>

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class ILayer {

public:
  virtual ~ILayer(){};
  virtual const MatrixXf activationUpdate(const MatrixXf &input) = 0;
  virtual MatrixXf activation(const MatrixXf &input) const = 0;
  virtual MatrixXf dActivation() = 0;
  virtual size_t inputWidth() const = 0;
  virtual size_t outputWidth() const = 0;
  virtual const MatrixXf &weights() const = 0;
  virtual void propagate(MatrixXf &error) = 0;
  virtual void propagate(ILayer &previous) = 0;
  virtual void applyGradient(float eps) = 0;
  virtual void print() const = 0;
  virtual const MatrixXf &delta() const = 0;
};

class Layer : public ILayer {
  friend class SigmoidLayer;
  friend class ReluLayer;

public:
  const MatrixXf activationUpdate(const MatrixXf &input);

  MatrixXf activation(const MatrixXf &input) const;

  size_t inputWidth() const { return _weights.rows(); }

  size_t outputWidth() const { return _weights.cols(); }

  const MatrixXf &weights() const { return _weights; }

  void propagate(MatrixXf &errors) {
    _delta = errors.cwiseProduct(dActivation());
  }

  void propagate(ILayer &previous) {
    MatrixXf preiousComponent =
        previous.delta() * previous.weights().transpose();
    _delta = preiousComponent.cwiseProduct(dActivation());
  }

  void applyGradient(float eps) {
    _bias -= (eps / delta().rows()) * delta().colwise().sum();
    MatrixXf weightGrad = _input.transpose() * delta();
    _weights -= (eps / weightGrad.rows()) * weightGrad;
  }

  void print() const;

  MatrixXf dActivation();

  const MatrixXf &delta() const { return _delta; }

private:
  Layer(std::function<float(float)> actFn, std::function<float(float)> dActFn,
        size_t level, const int inputWidth, const int outputWidth)
      : Layer(actFn, dActFn, level, MatrixXf::Random(inputWidth, outputWidth),
              RowVectorXf::Random(outputWidth)) {}

  Layer(std::function<float(float)> actFn, std::function<float(float)> dActFn,
        size_t level, const MatrixXf &weights, const RowVectorXf &bias)
      : _actFn(actFn), _dActFn(dActFn), _level(level), _weights(weights),
        _bias(bias) {}

  size_t _level;
  MatrixXf _input;
  MatrixXf _weights;
  RowVectorXf _bias;
  MatrixXf _currentZeds;
  MatrixXf _delta;
  std::function<float(float)> _actFn;
  std::function<float(float)> _dActFn;
};

class SigmoidLayer : public Layer {
public:
  SigmoidLayer(size_t level, const int inputWidth, const int outputWidth)
      : Layer(sigmoid, dSigmoid, level, inputWidth, outputWidth) {}

  SigmoidLayer(size_t level, const MatrixXf &weights, const RowVectorXf &bias)
      : Layer(sigmoid, dSigmoid, level, weights, bias) {}

  static float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
  static float dSigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
};

class ReluLayer : public Layer {
public:
  ReluLayer(size_t level, const int inputWidth, const int outputWidth)
      : Layer(relu, dRelu, level, inputWidth, outputWidth) {}

  ReluLayer(size_t level, const MatrixXf &weights, const RowVectorXf &bias)
      : Layer(relu, dRelu, level, weights, bias) {}

  static constexpr float LEAK_RATE = 0.001f;
  static float relu(float x) { return x > 0.0f ? x : LEAK_RATE * x; }
  static float dRelu(float x) { return x > 0.0f ? 1.0f : LEAK_RATE; }
};

#endif // LAYER_H