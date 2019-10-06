#include <iostream>
#include <simple_nnet/Layer.h>

float relu(float x) { return x > 0 ? x : 0; }

float dRelu(float x) { return x > 0 ? 1.0f : 0.0f; }

void ReluLayer::print() const {
  std::cout << "=== Layer " << _level << " ======" << std::endl;
  std::cout << "weights: " << std::endl << _weights << std::endl;
  std::cout << "bias: " << _bias << std::endl;
  std::cout << "====================" << std::endl << std::endl;
}

const MatrixXf &ReluLayer::activationUpdate(const MatrixXf &input) {
  _batchSize = input.rows();
  _input = input;
  _currentZeds = (input * _weights).rowwise() + _bias;
  _currentActivation = _currentZeds.unaryExpr(std::ref(relu));
  return _currentActivation;
}

MatrixXf ReluLayer::activation(const MatrixXf &input) const {
  MatrixXf zeds = (input * _weights).rowwise() + _bias;
  return zeds.unaryExpr(std::ref(relu));
}

MatrixXf ReluLayer::dActivation() {
  return _currentZeds.unaryExpr(std::ref(dRelu));
}