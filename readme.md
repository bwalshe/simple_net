# Background
If you're looking for a great deep learning framework, then this isn't it. This project started out as 
a bit of an experiment in using the [Eigen](http://eigen.tuxfamily.org) linear algebra library and 
trying to write a non-trivial C++ program, which is a language I don't use a lot in my day-to-day work.

# What does this do?
Once compiled this program will load [MNIST digit data](http://yann.lecun.com/exdb/mnist/) (or any 
files which use a similar format), perform one-hot encoding of the labels and then train a single hidden layer ANN against these data using back-prop. 

It'll display the error rate, and then attempt to classify the first 20 entries of the test set.

# Compiling and Running
## Requirements
* [Eigen](http://eigen.tuxfamily.org)
* zlib
* g++
* CMake 3

Currently this works with `g++` on Linux and WSL. I've tried getting it to work with Visual Studio 2017 / 
Windows 10, but zlib and endian coversions behave funny. `clang` should be fine, but I haven't tried it yet.

## Compiling
In the directory where you have cloned the repo, create a sub-directory `build`. Inside this 
sub-directory run `cmake`
```
$ mkdir build
$ cd build
$ cmake ..
```
This should produce an executable named `train`

## Running
To run this, you will need the [MNIST](http://yann.lecun.com/exdb/mnist/) data set. Assuming you have downloaded this into `~/data`, run
the training process as follows:
```
./train ~/data/t10k-images-idx3-ubyte.gz ~/data/t10k-labels-idx1-ubyte.gz
```

# Things that could be improved
* Add a softmax output layer.
* Give more control over the learning rate, number of layers, etc.
* ~~Make it easy to choose between ReLU and sigmoid activation functions.~~
* Annealing, adaptive learning rate, all the things that actually make neural nets viable.
* General refactor to allow the components to be tested.