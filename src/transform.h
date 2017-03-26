#ifndef PROJECTFASTPOINTS_TRANSFORM_H_H
#define PROJECTFASTPOINTS_TRANSFORM_H_H

#include "quaternion.h"

PointCloud<PointXYZI> ShiftPoints(PointCloud<PointXYZI> &h_cloud, std::vector<double> shift);

PointCloud<PointXYZI> RotatePoints(PointCloud<PointXYZI> &h_cloud, Quaternion &quaternion);

#endif //PROJECTFASTPOINTS_TRANSFORM_H_H
