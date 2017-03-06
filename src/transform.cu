#include "../src/point_cloud.h"

#include "transform.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void shifting(PointXYZI *d_points, double *d_shift) {

    int idx = threadIdx.x;
    d_points[idx].x += d_shift[0];
    d_points[idx].y += d_shift[1];
    d_points[idx].z += d_shift[2];
}

PointCloud shiftPoints(PointCloud &h_cloud, std::vector<double> shift) {

    PointXYZI *d_points;
    double *d_shift;

    PointXYZI *h_points = &(h_cloud.points[0]);
    double *h_shift = &(shift[0]);

    cudaMalloc((void**) &d_points, h_cloud.points.size() * sizeof(PointXYZI));
    cudaMalloc((void**) &d_shift, shift.size() * sizeof(double));

    cudaMemcpy(d_points, h_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, h_shift, shift.size() * sizeof(double), cudaMemcpyHostToDevice);

    int N = h_cloud.points.size();

    shifting<<<1,N>>>(d_points, d_shift);

    cudaMemcpy(h_points, d_points, h_cloud.points.size() * sizeof(PointXYZI), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shift, d_shift, shift.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_shift);

    return h_cloud;
}