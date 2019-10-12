#include <iostream>
#include <simple_nnet/Layer.h>

void Layer::print() const {
  std::cout << "=== Layer " << _level << " ======" << std::endl;
  std::cout << "weights: " << std::endl << _weights << std::endl;
  std::cout << "bias: " << _bias << std::endl;
  std::cout << "====================" << std::endl << std::endl;
}

const MatrixXf Layer::activationUpdate(const MatrixXf &input) {
  _input = input;
  _currentZeds = (input * _weights).rowwise() + _bias;
  return _currentZeds.unaryExpr(_actFn);
}

MatrixXf Layer::activation(const MatrixXf &input) const {
  MatrixXf zeds = (input * _weights).rowwise() + _bias;
  return zeds.unaryExpr(_actFn);
}

MatrixXf Layer::dActivation() { return _currentZeds.unaryExpr(_dActFn); }