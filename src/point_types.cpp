#include "point_types.h"

Point3D Point3D::operator+(Point3D &other) {
    return Point3D(this->x + other.x, this->y + other.y, this->z + other.z);
}

Point3D Point3D::operator-(Point3D &other) {
    return Point3D(this->x - other.x, this->y - other.y, this->z - other.z);
}

Point3D Point3D::operator+(Point3D &&other) {
    return Point3D(this->x + other.x, this->y + other.y, this->z + other.z);
}

Point3D Point3D::operator-(Point3D &&other) {
    return Point3D(this->x - other.x, this->y - other.y, this->z - other.z);
}

Point3D Point3D::operator*(double scalar) {
    return Point3D(this->x * scalar, this->y * scalar, this->z * scalar);
}

Point3D Point3D::operator/(double scalar) {
    return Point3D(this->x / scalar, this->y / scalar, this->z / scalar);
}

PointXYZI::PointXYZI(double x, double y, double z, double i) :
        Point3D(x,y,z), i(i)
{ }

std::ostream& operator<< (std::ostream &out, const PointXYZI &point)
{

 out << (float) point.x << " "
     << (float) point.y << " "
     << (float) point.z << " "
     << (float) point.i;

 return out;
}

std::string PointXYZI::SizeString() {
 return std::string("4 4 4 4");
}

std::string PointXYZI::TypeString() {
 return std::string("F F F F");
}

std::string PointXYZI::FieldsString() {
 return std::string("x y z i");
}

PointXYZINormal::PointXYZINormal(PointXYZI point) :
        PointXYZI(point)
{ }

PointXYZINormal::PointXYZINormal(PointXYZI point, double n_x, double n_y, double n_z) :
        PointXYZI(point), n_x(n_x), n_y(n_y), n_z(n_z)
{ }

PointXYZINormal::PointXYZINormal(double x, double y, double z, double i, double n_x, double n_y, double n_z) :
        PointXYZI(x,y,z,i), n_x(n_x), n_y(n_y), n_z(n_z)
{ }

std::string PointXYZINormal::SizeString() {
    return std::string("4 4 4 4 4 4 4");
}

std::string PointXYZINormal::TypeString() {
    return std::string("F F F F F F F");
}

std::string PointXYZINormal::FieldsString() {
    return std::string("x y z i normal_x normal_y normal_z");
}

std::ostream &operator<<(std::ostream &out, const PointXYZINormal &point) {

    out << (float) point.x   << " "
    << (float) point.y   << " "
    << (float) point.z   << " "
    << (float) point.i   << " "
    << (float) point.n_x << " "
    << (float) point.n_y << " "
    << (float) point.n_z;

    return out;
}
