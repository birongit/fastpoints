#include <float.h>
#include "kernels.h"
#include "../utils/utils.h"

typedef double (*func)(double n, double m);

__global__ void reduce(const Point3D * const d_points, const uint number,
                       Point3D *d_result, func* f, double id_element) {

    const int idx = threadIdx.x;
    const int size = blockDim.x;

    extern __shared__ Point3D d_reduce[];

    if (idx < number) {
        d_reduce[idx] = d_points[idx];
    } else {
        d_reduce[idx] = Point3D(id_element,id_element,id_element);
    }

    __syncthreads();

    for (unsigned int s = size / 2; s > 0; s >>= 1)
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

Point3D reduce(PointCloud<Point3D> &cloud, func& f, double identity_element) {

    Point3D *d_points;
    Point3D *d_result;

    auto num_points = cloud.points.size();

    auto size = next_power_of_two(num_points);

    func* h_f;
    func* d_f;
    h_f = (func*)malloc(sizeof(func));
    cudaMalloc((void**)&d_f,sizeof(func));
    cudaMemcpyFromSymbol( &h_f[0], f, sizeof(func));
    cudaMemcpy(d_f,h_f,sizeof(func),cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_result, sizeof(Point3D));
    cudaMalloc((void**) &d_points, sizeof(Point3D) * cloud.points.size());
    cudaMemcpy(d_points, &cloud.points[0], sizeof(Point3D) * cloud.points.size(), cudaMemcpyHostToDevice);

    int N = cloud.points.size();

    reduce<<<1, size, size*sizeof(Point3D)>>>(d_points, num_points, d_result, d_f, identity_element);

    cudaDeviceSynchronize();

    Point3D h_result;
    cudaMemcpy(&h_result, d_result, sizeof(Point3D), cudaMemcpyDeviceToHost);

    std::cout << "Result of kernel: " << h_result.x << " " << h_result.y << " " << h_result.z << std::endl;

    cudaFree(d_points);
    cudaFree(d_result);

    cudaFree(d_f);
    free(h_f);

    return h_result;

}

__device__ double max_d(double n, double m) {
    return (m > n) ? m : n;
}
__device__ func max_func = max_d;

double max_identity_element = - DBL_MAX;

Point3D reduce_max(PointCloud<Point3D> &cloud) {
    return reduce(cloud, max_func, max_identity_element);
}


__device__ double min_d(double n, double m) {
    return (m < n) ? m : n;
}
__device__ func min_func = min_d;

double min_identity_element = DBL_MAX;

Point3D reduce_min(PointCloud<Point3D> &cloud) {
    return reduce(cloud, min_func, min_identity_element);
}