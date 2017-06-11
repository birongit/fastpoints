#include "cuda_utils.h"
#include "../../../../../usr/local/cuda/include/cuda.h"

int get_cuda_version() {

    int version;
    cudaError_t cuda_err = cudaRuntimeGetVersion(&version);

    return (cuda_err == CUDA_SUCCESS) ? version : 0;
}

int get_driver_version() {

    int version;
    cudaError_t cuda_err = cudaDriverGetVersion(&version);

    return (cuda_err == CUDA_SUCCESS) ? version : 0;
}