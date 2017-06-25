#include "kernels.h"

typedef double (*func)(double n, double m);

__global__ void reduce(const Point3D * const d_points, Point3D *d_result, func* f, Point3D *d_reduce) {


    const int idx = threadIdx.x;


    d_reduce[idx] = d_points[idx];
    d_reduce[idx] = d_points[idx];

    __syncthreads();

    Point3D res;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
        {
            d_reduce[idx].x = (*f)(d_reduce[idx].x, d_reduce[idx + s].x);
            d_reduce[idx].y = (*f)(d_reduce[idx].y, d_reduce[idx + s].y);
            d_reduce[idx].z = (*f)(d_reduce[idx].z, d_reduce[idx + s].z);
        }
        __syncthreads();
    }

    if (idx == 0) {
        *d_result = d_reduce[idx];
    }

}

__device__ double max_d(double n, double m) {
    return (m > n) ? m : n;
}
__device__ func max_func = max_d;

void reduce_max(PointCloud<Point3D> &cloud) {

    Point3D *d_points;
    Point3D *d_result;
    Point3D *d_reduce;

    func* h_f;
    func* d_f;
    h_f = (func*)malloc(sizeof(func));
    cudaMalloc((void**)&d_f,sizeof(func));
    cudaMemcpyFromSymbol( &h_f[0], max_func, sizeof(func));
    cudaMemcpy(d_f,h_f,sizeof(func),cudaMemcpyHostToDevice);

    cudaMalloc(&d_reduce, sizeof(Point3D) * cloud.points.size());
    cudaMalloc((void**) &d_result, sizeof(Point3D));
    cudaMalloc((void**) &d_points, sizeof(Point3D) * cloud.points.size());
    cudaMemcpy(d_points, &cloud.points[0], sizeof(Point3D) * cloud.points.size(), cudaMemcpyHostToDevice);

    int N = cloud.points.size();

    reduce<<<1,N>>>(d_points, d_result, d_f, d_reduce);

    cudaDeviceSynchronize();

    Point3D h_result;
    cudaMemcpy(&h_result, d_result, sizeof(Point3D), cudaMemcpyDeviceToHost);

    std::cout << "Result of kernel: " << h_result.x << " " << h_result.y << " " << h_result.z << std::endl;

    cudaFree(d_points);
    cudaFree(d_result);
    cudaFree(d_reduce);

    cudaFree(d_f);
    free(h_f);

}