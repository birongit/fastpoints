#ifndef PROJECTFASTPOINTS_QUATERNION_H
#define PROJECTFASTPOINTS_QUATERNION_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include "../common/point_types.h"

class Quaternion {
public:

    CUDA_CALLABLE_MEMBER Quaternion() {};

    CUDA_CALLABLE_MEMBER ~Quaternion() {};

    CUDA_CALLABLE_MEMBER Quaternion(double x, double y, double z, double w): x(x), y(y), z(z), w(w) {};

    CUDA_CALLABLE_MEMBER Quaternion(PointXYZI &point): x(point.x), y(point.y), z(point.z), w(0.0) {};

    CUDA_CALLABLE_MEMBER Quaternion operator*(Quaternion other);

    CUDA_CALLABLE_MEMBER Quaternion operator*(double scalar);

    CUDA_CALLABLE_MEMBER Quaternion inverse();

    CUDA_CALLABLE_MEMBER Quaternion conjugate();

    CUDA_CALLABLE_MEMBER double norm();

    double x;
    double y;
    double z;
    double w;
};


#endif //PROJECTFASTPOINTS_QUATERNION_H
