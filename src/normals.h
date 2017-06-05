#ifndef FASTPOINTS_NORMALS_H
#define FASTPOINTS_NORMALS_H

#include "point_cloud.h"
#include "point_types.h"
#include <queue>
#include <float.h>
#include "geometry.h"

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
