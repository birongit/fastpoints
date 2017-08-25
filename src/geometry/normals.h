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

class Normals {

public:
    //Normals(PointCloud<Point3D> &cloud): cloud(cloud) {};
    Normals(PointCloud<Point3D> cloud): cloud(cloud) {};
    PointCloud<PointXYZINormal> estimate();

private:
    PointCloud<Point3D> find_neighbors_naive(Point3D &query_point, int k_neighbors);

    PointCloud<Point3D> cloud;
};


#endif //FASTPOINTS_NORMALS_H
