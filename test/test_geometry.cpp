#include "test_geometry.h"

int test_eig3() {

    std::vector<double> covar {0.00179001, -0.000598935, 0.000185897,
                               -0.000598935, 0.00173918, -0.000567059,
                               0.000185897, -0.000567059, 0.00050396};

    std::vector<double> correct_eigval {0.00250671, 0.00124485, 0.000281593};
    std::vector<double> correct_eigvec {0.657984, -0.70633, 0.261065,
                                        0.752237, 0.600606, -0.270946,
                                        0.0345799, 0.374661, 0.926517};

    std::vector<double> eigval(3);
    std::vector<double> eigvec(9);
    eigen3(&covar[0], &eigval[0], &eigvec[0]);

    assert(check_equal(eigval, correct_eigval, 0.00000001));
    assert(check_equal(eigvec, correct_eigvec, 0.000001));

    return 0;
}

int main(int argc, char * argv[]) {

    test_eig3();

    return 0;
}