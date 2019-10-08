#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <iostream>
#include <math.h> 

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

float sigmoid(float x);

float dSigmoid(float x);


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

class SigmoidLayer : public ILayer {

public:
  SigmoidLayer(size_t level, const int inputWidth, const int outputWidth) {
    _weights = MatrixXf::Random(inputWidth, outputWidth);
    _bias = RowVectorXf::Random(outputWidth);
    _level = level;
  }

  SigmoidLayer(size_t level, const MatrixXf &weights, const RowVectorXf &bias)
      : _weights(weights), _bias(bias), _level(level) {}

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
  size_t _level;
  MatrixXf _input;
  MatrixXf _weights;
  RowVectorXf _bias;
  MatrixXf _currentZeds;
  MatrixXf _delta;
};

#endif // LAYER_H