#include <iostream>
#include <simple_nnet/IdxLoader.h>

IdxContents::IdxContents(const gzFile &imageFile, const gzFile &labelFile) {
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
        Matrix<uint8_t, 1, Dynamic>::Map(data, imgSize).cast<float>() / 128);
  }
  _labels = std::make_unique<uint8_t[]>(_numImages);
  if (gzread(labelFile, _labels.get(), _numImages) != _numImages) {
    throw IdxLoadException("Failed to read label data.");
  }
}

void IdxContents::printSummary() const {
  std::cout << "Read " << numImages() << " images." << std::endl;
  std::cout << "Image size is " << _imgWidth << " by " << _imgHeight
            << " pixels." << std::endl
            << "There are " << numClasses() << " different lable values."
            << std::endl;
}

gzFile IdxContents::openFile(const char *path) {
  gzFile f = gzopen(path, "rb");
  if (f == NULL) {
    std::ostringstream os;
    throw IdxLoadException::gzipFail(path);
  }
  return f;
}

uint32_t IdxContents::readUint32(const gzFile &inFile) {
  uint32_t num = 0;
  int err;
  size_t len = sizeof(num);
  if (gzread(inFile, &num, len) != len) {
    throw IdxLoadException(gzerror(inFile, &err));
  }
  return be32toh(num);
}

void IdxContents::checkMagic(const gzFile &inFile, uint32_t expected) {
  uint32_t magic = readUint32(inFile);
  if (magic != expected) {
    throw IdxLoadException::badMagicNumber(magic, expected);
  }
}