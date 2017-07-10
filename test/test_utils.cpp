
#include <iostream>
#include "test_utils.h"

void test_next_power_of_two() {

    unsigned long i = 30;
    auto res_int = next_power_of_two(i);
    assert(res_int==32);

    unsigned long l = 2049;
    auto res_long = next_power_of_two(l);
    assert(res_long==4096);

    unsigned short s = 100;
    auto res_short = next_power_of_two(s);
    assert(res_short==128);

    unsigned int z = 0;
    auto res_zero = next_power_of_two(z);
    assert(res_zero==1);

}

int main(int argc, char * argv[]) {

    test_next_power_of_two();

    return 0;
}