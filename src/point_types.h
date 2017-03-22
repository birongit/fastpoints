#ifndef PROJECT_POINT_TYPES_H
#define PROJECT_POINT_TYPES_H


#include <iostream>

class PointXYZI {
public:
    PointXYZI() { };
    PointXYZI(double x, double y, double z, double i): x(x), y(y), z(z), i(i) { };
    ~PointXYZI() { };

    static std::string SizeString();
    static std::string TypeString();
    static std::string FieldsString();

    friend std::ostream& operator<< (std::ostream &out, const PointXYZI &point);

    double x,y,z,i;
};


#endif //PROJECT_POINT_TYPES_H
