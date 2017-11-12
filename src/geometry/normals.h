#ifndef FASTPOINTS_NORMALS_H
#define FASTPOINTS_NORMALS_H

#include "../common/point_cloud.h"
#include "../common/point_types.h"
#include <queue>
#include <float.h>
#include <algorithm>
#include "geometry.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T> class Normals {

public:
    Normals(PointCloud<T> cloud): cloud(cloud) {};
    PointCloud<PointXYZINormal> estimate();

private:
    PointCloud<T> find_neighbors_naive(T &query_point, int k_neighbors);

    void test(PointCloud<T> points, int size);

    PointCloud<T> cloud;
};

template<typename T>
PointCloud<PointXYZINormal> Normals<T>::estimate() {

    PointCloud<PointXYZINormal> normals;
    double eig_val[3], eig_vec[9];

    for (auto point : cloud.points) {

        auto neighbors = find_neighbors_naive(point, 10);
        auto covar = covariance(neighbors);

        eigen3(&covar[0], eig_val, eig_vec);

        normals.points.push_back(PointXYZINormal(point.x, point.y, point.z, 1.0, eig_vec[6], eig_vec[7], eig_vec[8]));
    }

    return normals;

}



#endif //FASTPOINTS_NORMALS_H
