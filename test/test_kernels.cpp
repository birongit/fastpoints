#include "test_kernels.h"

void test_reduce() {

    std::vector<Point3D> input{
            Point3D(-1.36545573e+00,  -1.05557640e+00,  -1.31559244e+00),
            Point3D(2.25514827e+00,  -6.37690115e-01,  -1.23241296e+00),
            Point3D(-6.50089035e-01,  -3.37900570e-01,   2.08824186e+00),
            Point3D(5.34672838e-01,  -1.44349888e+00,   9.11865657e-01),
            Point3D(1.13641539e+00,  -1.76824080e+00,   2.75086554e-01),
            Point3D(-2.53406386e+00,   4.08498688e-01,  -1.17000859e-01),
            Point3D(1.64803349e+00,   7.48866056e-01,  -8.21619790e-01),
            Point3D(8.06128916e-03,  -1.16117591e+00,   6.61879169e-01),
            Point3D(2.29689454e+00,  -3.14518292e-01,   1.30789968e+00),
            Point3D(1.71829808e+00,  -8.85344031e-02,  -1.71569931e+00),
            Point3D(5.33990084e-01,   7.68579287e-01,  -1.77831364e-01),
            Point3D(-1.04613525e+00,  -1.16619080e+00,  -1.29817385e+00),
            Point3D(-1.36614000e+00,  -9.74763252e-02,  -8.94043977e-01),
            Point3D(-2.61801057e+00,   1.54857110e+00,   8.31817521e-01),
            Point3D(2.90219237e-02,  -2.33534205e-01,  -5.03627218e-01),
            Point3D(-1.25713429e+00,  -2.58259549e-01,  -1.29504148e-01),
            Point3D(-6.54411017e-01,   2.19001469e+00,  -8.32638196e-01),
            Point3D(-3.86426046e-01,   2.12060712e+00,   2.21868096e+00),
            Point3D(6.35110596e-01,  -2.62931330e+00,  -9.73162641e-01),
            Point3D(-1.01919097e-01,   6.94788223e-01,  -7.96021309e-01)
    };

    PointCloud<Point3D> cloud(input);

    Point3D correct_min(-2.61801057e+00, -2.62931330e+00, -1.71569931e+00);
    Point3D correct_max( 2.29689454e+00,  2.19001469e+00,  2.21868096e+00);

    auto min = reduce_min(cloud);
    assert(check_equal(min, correct_min));
    auto max = reduce_max(cloud);
    assert(check_equal(max, correct_max));

}

void test_exclusive() {

    std::vector<double> input{
            0.14598861,  0.12668405,  0.54109701,  0.37165243,  0.92090129,
            0.69069576,  0.77604026,  0.87044717,  0.01266919,  0.87628207
    };

    std::vector<double> correct{
            0.        ,  0.14598861,  0.27267266,  0.81376967,  1.18542210,
            2.10632339,  2.79701915,  3.57305941,  4.44350658,  4.45617577
    };

    auto output = exclusive_scan(input);

    assert(check_equal(output, correct, 1.0e-8));

}

void test_inclusive() {

    std::vector<double> input{
            0.14598861,  0.12668405,  0.54109701,  0.37165243,  0.92090129,
            0.69069576,  0.77604026,  0.87044717,  0.01266919,  0.87628207
    };

    std::vector<double> correct{
            0.14598861,  0.27267266,  0.81376967,  1.1854221 ,  2.10632339,
            2.79701915,  3.57305941,  4.44350659,  4.45617578,  5.33245785
    };

    auto output = inclusive_scan(input);

    assert(check_equal(output, correct, 1.0e-8));

}

int main(int argc, char * argv[]) {


    test_reduce();

    test_exclusive();

    test_inclusive();

    return 0;
}