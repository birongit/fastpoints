#ifndef FASTPOINTS_TEST_UTILS_H
#define FASTPOINTS_TEST_UTILS_H

#include <vector>
#include <math.h>
#include <assert.h>
#include "../src/common/point_types.h"

bool check_equal(std::vector<double>& val1, std::vector<double>& val2, float tolerance = 0.0f);
bool check_equal(Point3D& val1, Point3D& val2, float tolerance = 0.0f);


#endif //PROJECTFASTPOINTS_TEST_UTILS_H
