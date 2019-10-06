cmake_minimum_required (VERSION 3.10)
project (simple_nnnet VERSION 1.0
                DESCRIPTION "A basic implementation of a neural network"
                LANGUAGES CXX)
include(FetchContent)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ZLIB)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    add_subdirectory(test)
endif()