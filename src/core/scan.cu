#include "kernels.h"

//TODO scan kernels


std::vector<double> exclusive_scan(std::vector<double> &in) {

    return std::vector<double>();
}

__global__ void inclusive(const double *d_in,
                                double* d_out,
                                size_t size,
                                int inc) {

    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    if (myId < size) {

        if (myId >= inc) {
            d_out[myId] = d_in[myId] + d_in[myId - inc];
        } else {
            d_out[myId] = d_in[myId];
        }
    }

}

std::vector<double> inclusive_scan(std::vector<double> &in) {

    double *d_in, *d_out;
    std::vector<double> h_out(in.size());
    cudaMalloc((void**)&d_in, sizeof(double) * in.size());
    cudaMalloc((void**)&d_out, sizeof(double) * in.size());

    cudaMemcpy(&d_in[0], &in[0], sizeof(double) * in.size(), cudaMemcpyHostToDevice);

    int N = in.size();

    int inc = 1;
    while(inc <= in.size()) {

        inclusive<<<1, N>>>(d_in, d_out, in.size(), inc);

        cudaDeviceSynchronize();

        cudaMemcpy(d_in, d_out, sizeof(double) * in.size(), cudaMemcpyDeviceToDevice);

        inc *= 2;
    }

    cudaMemcpy(&h_out[0], d_out, sizeof(double) * in.size(), cudaMemcpyDeviceToHost);

    return h_out;
}