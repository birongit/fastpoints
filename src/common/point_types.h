#ifndef PROJECT_POINT_TYPES_H
#define PROJECT_POINT_TYPES_H

#include <iostream>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Point3D {
public:
    CUDA_CALLABLE_MEMBER Point3D() { };
    CUDA_CALLABLE_MEMBER Point3D(double x, double y, double z) : x(x), y(y), z(z) { };
    CUDA_CALLABLE_MEMBER ~Point3D() { };

    CUDA_CALLABLE_MEMBER Point3D operator+(Point3D &other);
    CUDA_CALLABLE_MEMBER Point3D operator-(Point3D &other);

    CUDA_CALLABLE_MEMBER Point3D operator+(Point3D &&other);
    CUDA_CALLABLE_MEMBER Point3D operator-(Point3D &&other);

    CUDA_CALLABLE_MEMBER Point3D operator*(double scalar);
    CUDA_CALLABLE_MEMBER Point3D operator/(double scalar);

    CUDA_CALLABLE_MEMBER double distance2(Point3D &other);

    double x,y,z;
};

class PointXYZI : public Point3D {
public:
    PointXYZI() { };
    PointXYZI(double x, double y, double z, double i);
    PointXYZI(Point3D point);
    virtual ~PointXYZI() { };

    static std::string SizeString();
    static std::string TypeString();
    static std::string FieldsString();

    friend std::ostream& operator<< (std::ostream &out, const PointXYZI &point);

    double i;
};

class PointXYZINormal : public PointXYZI {
public:
    PointXYZINormal() { };
    PointXYZINormal(PointXYZI point);
    PointXYZINormal(Point3D point);
    PointXYZINormal(PointXYZI point, double n_x, double n_y, double n_z);
    PointXYZINormal(double x, double y, double z, double i, double n_x, double n_y, double n_z);
    ~PointXYZINormal() { };

    static std::string SizeString();
    static std::string TypeString();
    static std::string FieldsString();

    friend std::ostream& operator<< (std::ostream &out, const PointXYZINormal &point);

    double n_x, n_y, n_z;
};


#endif //PROJECT_POINT_TYPES_H
