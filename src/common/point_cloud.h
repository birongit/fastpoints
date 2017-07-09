#ifndef FASTPOINTS_POINT_CLOUD_H
#define FASTPOINTS_POINT_CLOUD_H

#include "point_types.h"
#include <vector>

template <class T> class PointCloud {
public:
    PointCloud() {};
    PointCloud(std::vector<T> points): points(points) {};
    std::vector<T> points;

};


#endif //FASTPOINTS_POINT_CLOUD_H
