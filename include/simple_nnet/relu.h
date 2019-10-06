#ifndef RELU_H
#define RELU_H

float relu(float x) { return x > 0 ? x : 0; }

float dRelu(float x) { return x > 0 ? 1.0f : 0.0f; }

#endif // RELU_H