#ifndef IMAGE_SAMPLER_H
#define IMAGE_SAMPLER_H
#include <vector>
#include<memory>

#include "IdxLoader.h"


using Eigen::MatrixXf;
using Eigen::RowVectorXf;

class OneHotEncoder {
public:
  OneHotEncoder(uint max) { _rows = MatrixXf::Identity(max, max); }

  RowVectorXf encode(uint x) { return _rows.row(x); }

private:
  MatrixXf _rows;
};

class ImageSampler {
public:
  ImageSampler(const char *imagePath, const char *labelPath) {
    data = std::make_unique<IdxContents>(imagePath, labelPath);
  }


  uint32_t inputWidth() const { return data->imageWidth() * data->imageHeight(); }

  uint8_t targetWidth() const { return data->numClasses(); }

  size_t totalImages() const { return data->numImages(); }

  std::pair<MatrixXf, MatrixXf> nextSample(size_t count);

private:
  std::unique_ptr<IdxContents> data;
  std::vector<size_t> _indexes;

  size_t nextRandomIndex();
  void resetRandomIndexes();
};

#endif // IMAGE_SAMPLER_H