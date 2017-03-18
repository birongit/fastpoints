//
// Created by Birgit Henke on 3/17/17.
//

#include <math.h>
#include "quaternion.h"

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator*(Quaternion other) {

    double w = this->w * other.w
               - this->x * other.x
               - this->y * other.y
               - this->z * other.z;

    double x = this->w * other.x
               + this->x * other.w
               + this->y * other.z
               - this->z * other.y;

    double y = this->w * other.y
               + this->y * other.w
               + this->z * other.x
               - this->x * other.z;

    double z = this->w * other.z
               + this->z * other.w
               + this->x * other.y
               - this->y * other.x;

    return Quaternion(x, y, z, w);
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::inverse() {
    return conjugate() * (1.0/norm());
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::conjugate() {
    return Quaternion(-x, -y, -z, w);
}

CUDA_CALLABLE_MEMBER double Quaternion::norm() {
    return sqrt(x * x + y * y + z * z + w * w);
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator*(double scalar) {
    return Quaternion(x*scalar, y*scalar, z*scalar, w*scalar);
}
