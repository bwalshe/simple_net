#include <iostream>
#include <simple_nnet/Layer.h>

float sigmoid(float x) { 
    return 1.0f / (1.0f + exp(-x)); 
}

float dSigmoid(float x) { 
    return sigmoid(x)*(1-sigmoid(x));
}

void SigmoidLayer::print() const {
  std::cout << "=== Layer " << _level << " ======" << std::endl;
  std::cout << "weights: " << std::endl << _weights << std::endl;
  std::cout << "bias: " << _bias << std::endl;
  std::cout << "====================" << std::endl << std::endl;
}

const MatrixXf SigmoidLayer::activationUpdate(const MatrixXf &input) {
  _input = input;
  _currentZeds = (input * _weights).rowwise() + _bias;
  return _currentZeds.unaryExpr(std::ref(sigmoid));
}

MatrixXf SigmoidLayer::activation(const MatrixXf &input) const {
  MatrixXf zeds = (input * _weights).rowwise() + _bias;
  return zeds.unaryExpr(std::ref(sigmoid));
}

MatrixXf SigmoidLayer::dActivation() {
  return _currentZeds.unaryExpr(std::ref(dSigmoid));
}