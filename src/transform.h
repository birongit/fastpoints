#ifndef PROJECTFASTPOINTS_TRANSFORM_H_H
#define PROJECTFASTPOINTS_TRANSFORM_H_H

#include "quaternion.h"

PointCloud ShiftPoints(PointCloud &h_cloud, std::vector<double> shift);

PointCloud RotatePoints(PointCloud &h_cloud, Quaternion &quaternion);

#endif //PROJECTFASTPOINTS_TRANSFORM_H_H
