#include <simple_nnet/ImageSampler.h>

size_t ImageSampler::nextRandomIndex() {
  if (_indexes.size() == 0)
    resetRandomIndexes();
  auto out = _indexes.back();
  _indexes.pop_back();
  return out;
}

void ImageSampler::resetRandomIndexes() {
  _indexes.clear();
  for (size_t i = 0; i < data->numImages(); ++i) {
    _indexes.push_back(i);
  }
  std::random_shuffle(_indexes.begin(), _indexes.end());
}

std::pair<MatrixXf, MatrixXf> ImageSampler::nextSample(size_t count) {
  MatrixXf images(count, inputWidth());
  MatrixXf labels(count, targetWidth());
  OneHotEncoder encoder = OneHotEncoder(targetWidth());

  for (size_t i = 0; i < count; ++i) {
    auto randIndex = nextRandomIndex();
    images.row(i) = data->image(randIndex);
    labels.row(i) = encoder.encode(data->label(randIndex));
  }
  return std::pair<MatrixXf, MatrixXf>(images, labels);
}