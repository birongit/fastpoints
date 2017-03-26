#ifndef PROJECT_POINT_TYPES_H
#define PROJECT_POINT_TYPES_H


#include <iostream>

class PointXYZI {
public:
    PointXYZI() { };
    PointXYZI(double x, double y, double z, double i);
    virtual ~PointXYZI() { };

    static std::string SizeString();
    static std::string TypeString();
    static std::string FieldsString();

    friend std::ostream& operator<< (std::ostream &out, const PointXYZI &point);

    double x,y,z,i;
};

class PointXYZINormal : public PointXYZI {
public:
    PointXYZINormal() { };
    PointXYZINormal(PointXYZI point);
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
