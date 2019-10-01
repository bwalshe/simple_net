#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>

#include "BasicNeuralNet.h"
#include "ImageSampler.h"

#define BATCH_SIZE 100
#define NUM_BATCHES 1000

int mod2(int x) { return abs(x) % 2; }

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Useage: " << argv[0] << " [image file] [class file]"
              << std::endl;
    return 1;
  }

  srand((unsigned int)time(NULL));

  ImageSampler images(argv[1], argv[2]);

  Network net(images.inputWidth(), 100);
  net.addLayer(images.targetWidth());

  for (auto i = 0; i < NUM_BATCHES; ++i) {
    std::pair<MatrixXf, MatrixXf> sample = images.nextSample(BATCH_SIZE);
    MatrixXf input = sample.first;
    MatrixXf target = sample.second;
    auto error = net.backprop(input, target, 0.000005f);
    if (i % (NUM_BATCHES / 10) == 0)
      std::cout << "Error [" << i << "] = " << error << std::endl;
  }
}
