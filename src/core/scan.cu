#include "kernels.h"

__global__ void inclusive(const double *d_in,
                          double* d_out,
                          size_t size,
                          int inc) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {

        if (idx >= inc) {
            d_out[idx] = d_in[idx] + d_in[idx - inc];
        } else {
            d_out[idx] = d_in[idx];
        }
    }

}

__global__ void subtract(double *d_scan,
                         double* d_diff,
                         size_t size) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {
        d_scan[idx] = d_scan[idx] - d_diff[idx];
    }
}

void inclusive_gpu(double *d_in, double *d_out, size_t size) {

    int inc = 1;
    while(inc <= size) {

        inclusive<<<1, size>>>(d_in, d_out, size, inc);

        cudaDeviceSynchronize();

        cudaMemcpy(d_in, d_out, sizeof(double) * size, cudaMemcpyDeviceToDevice);

        inc *= 2;
    }

}

std::vector<double> exclusive_scan(std::vector<double> &in) {

    double *d_in, *d_out, *d_diff;
    std::vector<double> h_out(in.size());

    cudaMalloc((void**)&d_in, sizeof(double) * in.size());
    cudaMalloc((void**)&d_out, sizeof(double) * in.size());
    cudaMalloc((void**)&d_diff, sizeof(double) * in.size());

    cudaMemcpy(&d_in[0], &in[0], sizeof(double) * in.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_diff[0], &d_in[0], sizeof(double) * in.size(), cudaMemcpyDeviceToDevice);

    inclusive_gpu(d_in, d_out, in.size());

    subtract<<<1, in.size()>>>(d_out, d_diff, in.size());

    cudaMemcpy(&h_out[0], d_out, sizeof(double) * in.size(), cudaMemcpyDeviceToHost);

    return h_out;

}

std::vector<double> inclusive_scan(std::vector<double> &in) {

    double *d_in, *d_out;
    std::vector<double> h_out(in.size());

    cudaMalloc((void**)&d_in, sizeof(double) * in.size());
    cudaMalloc((void**)&d_out, sizeof(double) * in.size());

    cudaMemcpy(&d_in[0], &in[0], sizeof(double) * in.size(), cudaMemcpyHostToDevice);

    inclusive_gpu(d_in, d_out, in.size());

    cudaMemcpy(&h_out[0], d_out, sizeof(double) * in.size(), cudaMemcpyDeviceToHost);

    return h_out;
}