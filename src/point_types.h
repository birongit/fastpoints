#ifndef PROJECT_POINT_TYPES_H
#define PROJECT_POINT_TYPES_H


class PointXYZI {
public:
    PointXYZI() { };
    PointXYZI(double x, double y, double z, double i): x(x), y(y), z(z), i(i) { };
    ~PointXYZI() { };

    double x,y,z,i;
};


#endif //PROJECT_POINT_TYPES_H
