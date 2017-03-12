project(projectfastpoints)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 14)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)

add_subdirectory(src)
add_subdirectory(examples)

