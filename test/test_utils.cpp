#include "test_utils.h"

bool check_equal(std::vector<double>& val1, std::vector<double>& val2, float tolerance = 0.0f) {

    if (val1.size() != val2.size()) return false;

    for (int i = 0; i < val1.size(); i++) {
        if (fabs(val2[i] - val1[i]) > tolerance) return false;
    }

    return true;

}