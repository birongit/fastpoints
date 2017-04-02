#ifndef PROJECTFASTPOINTS_GEOMETRY_H
#define PROJECTFASTPOINTS_GEOMETRY_H

#include "point_types.h"
#include "point_cloud.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template<typename T> CUDA_CALLABLE_MEMBER Point3D mean(T *d_points, int n) {

    if (n == 0) return Point3D();

    Point3D mean(0,0,0);

    for (int i = 0; i < n; i++) {
        mean = mean + static_cast<Point3D>(d_points[i]);
    }

    mean = mean / n;

    return mean;
};


#endif //PROJECTFASTPOINTS_GEOMETRY_H
