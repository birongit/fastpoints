#include "test_utils.h"

bool check_equal(std::vector<double>& val1, std::vector<double>& val2, float tolerance) {

    if (val1.size() != val2.size()) return false;

    for (int i = 0; i < val1.size(); i++) {
        if (fabs(val2[i] - val1[i]) > tolerance) return false;
    }

    return true;

}

bool check_equal(Point3D& val1, Point3D& val2, float tolerance) {

    if (fabs(val1.x - val2.x) > tolerance) return false;
    if (fabs(val1.y - val2.y) > tolerance) return false;
    if (fabs(val1.z - val2.z) > tolerance) return false;

    return true;

}