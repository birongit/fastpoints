#ifndef PROJECTFASTPOINTS_GEOMETRY_H
#define PROJECTFASTPOINTS_GEOMETRY_H

#include "../common/point_types.h"
#include "../common/point_cloud.h"
#include <assert.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// matrix must be 3x3 and symmetric
CUDA_CALLABLE_MEMBER void eigen3(double* matrix, double* eig_val, double* eig_vec);

CUDA_CALLABLE_MEMBER double det3(double *matrix);

CUDA_CALLABLE_MEMBER void norm(double *vector, int size);

template<typename T> CUDA_CALLABLE_MEMBER Point3D mean(T *points, int n) {

    if (n == 0) return Point3D();

    Point3D mean(0,0,0);

    for (int i = 0; i < n; i++) {
        mean = mean + static_cast<Point3D>(points[i]);
    }

    mean = mean / n;

    return mean;
};

template<typename T> CUDA_CALLABLE_MEMBER void covariance(T *points, int n, double* covariance) {

    if (n == 0) return;

    Point3D mean(0,0,0);

    // TODO use elements of covariance
    double xx = 0; //covariance[0]
    double yy = 0; //covariance[4]
    double zz = 0; //covariance[8]
    double xy = 0; //covariance[1]
    double xz = 0; //covariance[2]
    double yz = 0; //covariance[5]

    for (int i = 0; i < n; i++) {

        mean = mean + static_cast<Point3D>(points[i]);

        xx += points[i].x * points[i].x;
        yy += points[i].y * points[i].y;
        zz += points[i].z * points[i].z;

        xy += points[i].x * points[i].y;
        yz += points[i].y * points[i].z;
        xz += points[i].x * points[i].z;
    }

    mean = mean / n;

    covariance[0] = xx / n - mean.x * mean.x;
    covariance[1] = xy / n - mean.x * mean.y;
    covariance[2] = xz / n - mean.x * mean.z;

    covariance[3] = covariance[1];
    covariance[4] = yy / n - mean.y * mean.y;
    covariance[5] = yz / n - mean.y * mean.z;

    covariance[6] = covariance[2];
    covariance[7] = covariance[5];
    covariance[8] = zz / n - mean.z * mean.z;

}

template<typename T> std::vector<double> covariance(PointCloud<T> cloud) {

    std::vector<double> covar(9, 0.0);
    covariance(&cloud.points[0], cloud.points.size(), &covar[0]);

    return covar;
}

template<typename T> Point3D mean(PointCloud<T> cloud) {

    Point3D mean_point = mean(&cloud.points[0], cloud.points.size());

    return mean_point;
}


#endif //PROJECTFASTPOINTS_GEOMETRY_H
