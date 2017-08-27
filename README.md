# Fastpoints

**Fast 3D point cloud processing lib based on C++/cuda**

This repository shows current work in progress. It is intended to become a simple and lightweight library for fast GPU-based point cloud processing with minimal dependencies.

## Dependencies

- CMake
- gcc or clang
- CUDA

## Build

Clone the repository:
```
git clone https://github.com/birongit/fastpoints.git
cd fastpoints
```
   
Build the project:
```
mkdir build && cd build
cmake ..
make -j8
make test
```
Demo of currently supported features can be found in `examples`.
