#ifndef IMAGE_SAMPLER_H
#define IMAGE_SAMPLER_H
#include <algorithm>
#include <utility>
#include <vector>

#include "IdxLoader.h"

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::RowVectorXf;


class ImageSampler {
public:
  ImageSampler(const char *imagePath, const char *labelPath) {
    data = IdxContents::fromPath(imagePath, labelPath);
  }

  ~ImageSampler() { delete data; }

  uint32_t inputWidth() { return data->imageWidth() * data->imageHeight(); }

  uint8_t targetWidth() { return data->numClasses(); }

  std::pair<MatrixXf, MatrixXf> nextSample(size_t count) {
    MatrixXf images(count, inputWidth());
    MatrixXf labels(count, targetWidth());
    MatrixXf oneHotRows = MatrixXf::Identity(targetWidth(), targetWidth());

    for (size_t i = 0; i < count; ++i) {
      auto randIndex = nextRandomIndex();
      images.row(i) = data->image(randIndex);
      labels.row(i) = oneHotRows.row(data->label(randIndex));
    }
    return std::pair<MatrixXf, MatrixXf>(images, labels);
  }

private:
  IdxContents *data;
  std::vector<size_t> _indexes;

  size_t nextRandomIndex() {
    if (_indexes.size() == 0)
      resetRandomIndexes();
    auto out = _indexes.back();
    _indexes.pop_back();
    return out;
  }
  void resetRandomIndexes() {
    _indexes.clear();
    for (size_t i = 0; i < data->numImages(); ++i) {
      _indexes.push_back(i);
    }
    std::random_shuffle(_indexes.begin(), _indexes.end());
  }
};

#endif // IMAGE_SAMPLER_H