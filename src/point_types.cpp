#include "point_types.h"

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
