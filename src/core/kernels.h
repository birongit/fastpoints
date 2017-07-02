#ifndef FASTPOINTS_KERNELS_H
#define FASTPOINTS_KERNELS_H

#include "../common/point_cloud.h"

void reduce_max(PointCloud<Point3D> &cloud);

void reduce_min(PointCloud<Point3D> &cloud);

#endif //FASTPOINTS_KERNELS_H
