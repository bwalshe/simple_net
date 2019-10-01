#ifndef IDX_LOADER_H
#define IDX_LOADER_H

#include <Eigen/Dense>
#include <algorithm>
#include <endian.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include <zlib.h>
}

#define IMAGE_MAGIC 2051
#define LABEL_MAGIC 2049

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowVectorXf;

uint32_t readUint32(const gzFile &file);

struct IdxLoadException : public std::exception {
  IdxLoadException(const char *reason) { _reason += reason; }
  virtual const char *what() const throw() { return _reason.c_str(); }

  static IdxLoadException badMagicNumber(uint32_t number, uint32_t expected) {
    std::ostringstream os;
    os << "Bad idx magic number, was expecting " << expected << ", but found "
       << number;
    return IdxLoadException(os);
  }

  static IdxLoadException imageFail(size_t imageNumber, size_t total) {
    std::ostringstream os;
    os << "Failed to read image " << imageNumber << " of " << total;
    return IdxLoadException(os);
  }

  static IdxLoadException gzipFail(const char *path) {
    std::ostringstream os;
    os << "Failed to open " << path;
    return IdxLoadException(os);
  }

private:
  std::string _reason;
  IdxLoadException(const std::ostringstream &os)
      : IdxLoadException(os.str().c_str()) {}
};

struct LabeledImage {
  RowVectorXf image;
  uint8_t label;
};

struct IdxContents {

  ~IdxContents() { delete _labels; }

  int numImages() { return _images.size(); }
  uint8_t numClasses() {
    return *std::max_element(_labels, _labels + numImages()) + 1;
  }
  int imageWidth() { return _imgWidth; }
  int imageHeight() { return _imgHeight; }

  RowVectorXf image(const size_t i) { return _images.at(i); }
  LabeledImage labelledImage(const size_t i) { return {image(i), label(i)}; }

  uint8_t label(const size_t i) { return _labels[i]; }

  void printSummary() {
    std::cout << "Read " << numImages() << " images." << std::endl;
    std::cout << "Image size is " << _imgWidth << " by " << _imgHeight
              << " pixels." << std::endl
              << "There are " << numClasses() << " different lable values."
              << std::endl;
  }

  static IdxContents *fromPath(const char *imagePath, const char *labelPath) {
    gzFile imageFile, labelFile;
    imageFile = openFile(imagePath);
    labelFile = openFile(labelPath);

    return new IdxContents(imageFile, labelFile);
  }

private:
  size_t _numImages;
  size_t _imgWidth;
  size_t _imgHeight;
  std::vector<RowVectorXf> _images;
  uint8_t *_labels;

  IdxContents(const gzFile &imageFile, const gzFile &labelFile) {
    checkMagic(imageFile, IMAGE_MAGIC);
    checkMagic(labelFile, LABEL_MAGIC);
    _numImages = readUint32(imageFile);
    size_t numLabels = readUint32(labelFile);
    if (_numImages != numLabels) {
      throw IdxLoadException(
          "Image file and lable file do not have the same number of entries.");
    }
    _imgWidth = readUint32(imageFile);
    _imgHeight = readUint32(imageFile);
    size_t imgSize = _imgWidth * _imgHeight;
    for (int i = 0; i < _numImages; ++i) {
      uint8_t *data = new uint8_t[imgSize];
      if (gzread(imageFile, data, imgSize) != imgSize) {
        throw IdxLoadException::imageFail(i, _numImages);
      }
      _images.push_back(
          Matrix<uint8_t, 1, Dynamic>::Map(data, imgSize).cast<float>() / 256);
    }
    _labels = new uint8_t[_numImages];
    if (gzread(labelFile, _labels, _numImages) != _numImages) {
      throw IdxLoadException("Failed to read label data.");
    }
  }

  static void checkMagic(const gzFile &inFile, uint32_t expected) {
    uint32_t magic = readUint32(inFile);
    if (magic != expected) {
      throw IdxLoadException::badMagicNumber(magic, expected);
    }
  }

  static gzFile openFile(const char *path) {
    gzFile f = gzopen(path, "rb");
    if (f == NULL) {
      std::ostringstream os;
      throw IdxLoadException::gzipFail(path);
    }
    return f;
  }
};

uint32_t readUint32(const gzFile &inFile) {
  uint32_t num = 0;
  int err;
  size_t len = sizeof(num);
  if (gzread(inFile, &num, len) != len) {
    fprintf(stderr, "gzread err: %s\n", gzerror(inFile, &err));
    exit(1);
  }
  return be32toh(num);
}

#endif // IDX_LOADER_H