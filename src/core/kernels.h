#ifndef FASTPOINTS_KERNELS_H
#define FASTPOINTS_KERNELS_H

#include "../common/point_cloud.h"

Point3D reduce_max(PointCloud<Point3D> &cloud);

Point3D reduce_min(PointCloud<Point3D> &cloud);

std::vector<double> exclusive_scan(std::vector<double> &in);

std::vector<double> inclusive_scan(std::vector<double> &in);

#endif //FASTPOINTS_KERNELS_H
