#include "../common/point_cloud.h"

#include "transform.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 512;

__global__ void shifting(PointXYZI *d_points, double *d_shift, size_t size) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_points[idx].x += d_shift[0];
        d_points[idx].y += d_shift[1];
        d_points[idx].z += d_shift[2];
    }
}

__global__ void rotate(PointXYZI *d_points, Quaternion *d_quaternion, size_t size) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    Quaternion q(d_points[idx]);

    Quaternion q_prime = *d_quaternion * q * d_quaternion->inverse();

    d_points[idx].x = q_prime.x;
    d_points[idx].y = q_prime.y;
    d_points[idx].z = q_prime.z;

}


PointCloud<PointXYZI> shift_points(PointCloud<PointXYZI> &h_cloud, std::vector<double> shift) {

    PointXYZI *d_points;
    double *d_shift;

    PointXYZI *h_points = &(h_cloud.points[0]);
    double *h_shift = &(shift[0]);

    cudaMalloc((void**) &d_points, h_cloud.points.size() * sizeof(PointXYZI));
    cudaMalloc((void**) &d_shift, shift.size() * sizeof(double));

    cudaMemcpy(d_points, h_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, h_shift, shift.size() * sizeof(double), cudaMemcpyHostToDevice);

    long N = h_cloud.points.size();
    int NUM_BLOCKS = ceil(float(N) / THREADS_PER_BLOCK);

    shifting<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_points, d_shift, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_points, d_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shift, d_shift, shift.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_shift);

    return h_cloud;
}

PointCloud<PointXYZI> rotate_points(PointCloud<PointXYZI> &h_cloud, Quaternion &h_quaternion){

    PointXYZI *d_points;
    PointXYZI *h_points = &(h_cloud.points[0]);

    Quaternion *d_quaternion;

    cudaMalloc((void**) &d_points, h_cloud.points.size() * sizeof(PointXYZI));
    cudaMalloc((void**) &d_quaternion, sizeof(Quaternion));

    cudaMemcpy(d_points, h_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quaternion, &h_quaternion, sizeof(Quaternion), cudaMemcpyHostToDevice);

    long N = h_cloud.points.size();
    int NUM_BLOCKS = ceil(float(N) / THREADS_PER_BLOCK);

    rotate<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_points, d_quaternion, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_points, d_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_quaternion);

    return h_cloud;

}