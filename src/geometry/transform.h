#ifndef FASTPOINTS_TRANSFORM_H
#define FASTPOINTS_TRANSFORM_H

#include "quaternion.h"

PointCloud<PointXYZI> shift_points(PointCloud<PointXYZI> &h_cloud, std::vector<double> shift);

PointCloud<PointXYZI> rotate_points(PointCloud<PointXYZI> &h_cloud, Quaternion &quaternion);

#endif //FASTPOINTS_TRANSFORM_H
