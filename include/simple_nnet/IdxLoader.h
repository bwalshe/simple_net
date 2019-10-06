#ifndef IDX_LOADER_H
#define IDX_LOADER_H

#include <Eigen/Dense>
#include <sstream>
#include <vector>
#include <memory>

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

struct IdxContents {

  IdxContents(const char *imagePath, const char *labelPath)
      : IdxContents(openFile(imagePath), openFile(labelPath)) {}

  IdxContents(const gzFile &imageFile, const gzFile &labelFile);

  int numImages() const { return _images.size(); }
  uint8_t numClasses() const {
    return *std::max_element(_labels.get(), _labels.get() + numImages()) + 1;
  }
  int imageWidth() const { return _imgWidth; }
  int imageHeight() const { return _imgHeight; }

  RowVectorXf image(const size_t i) const { return _images.at(i); }
  uint8_t label(const size_t i) const { return _labels[i]; }

  void printSummary() const;

private:
  size_t _numImages;
  size_t _imgWidth;
  size_t _imgHeight;
  std::vector<RowVectorXf> _images;
  std::unique_ptr<uint8_t[]> _labels;

  static void checkMagic(const gzFile &inFile, uint32_t expected);
  static gzFile openFile(const char *path);
  static inline uint32_t readUint32(const gzFile &inFile);
};

#endif // IDX_LOADER_H