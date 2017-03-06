project(projectfastpoints)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)
add_subdirectory(examples)

