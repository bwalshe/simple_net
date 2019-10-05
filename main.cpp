#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>

#include "BasicNeuralNet.h"
#include "IdxLoader.h"
#include "ImageSampler.h"

#define BATCH_SIZE 100
#define NUM_BATCHES 10000
#define HIDDEN_LAYER_SIZE 10
#define GRADIENT_DELTA 0.00001f
#define DELTA_DECAY 0.6f
#define TEST_SIZE 10

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Useage: " << argv[0]
              << " [image file] [lable file] [test file] [teat label file]"
              << std::endl;
    return 1;
  }

  srand((unsigned int)time(NULL));

  ImageSampler trainImages(argv[1], argv[2]);
  IdxContents testImages(argv[3], argv[4]);

  std::cout << "Loaded " << trainImages.totalImages() << " images" 
            << " with " << (int) trainImages.targetWidth() << " classes" << std::endl;

  Network net(trainImages.inputWidth(), HIDDEN_LAYER_SIZE);
  net.addLayer(trainImages.targetWidth());

  float delta = GRADIENT_DELTA;
  for (auto i = 0; i < NUM_BATCHES; ++i) {
    std::pair<MatrixXf, MatrixXf> sample = trainImages.nextSample(BATCH_SIZE);
    MatrixXf input = sample.first;
    MatrixXf target = sample.second;
    auto error = net.backprop(input, target, GRADIENT_DELTA);
    if (i > 0 && (i % (NUM_BATCHES / 10) == 0)) {
      std::cout << "Error [" << i << "] = " << error << std::endl;
      delta *= DELTA_DECAY;
      std::cout << "Reducing delta to " << delta << std::endl;
    }
  }

  OneHotEncoder encoder = OneHotEncoder(10);
  for (auto i = 0; i < TEST_SIZE; ++i) {
   
    MatrixXf result = net.feed(testImages.image(i));
    MatrixXf expected = encoder.encode(testImages.label(i));
    MatrixXf errors = expected - result;
    std::cout << "Label: " << (int)testImages.label(i)
              << ", expected: " << expected
              << std::endl;
    std::cout << "Output: " << result << std::endl; 
    std::cout << "Errors: " << errors << std::endl;
    std::cout << "Squared norm = " << errors.squaredNorm() << std::endl;
  }
}
