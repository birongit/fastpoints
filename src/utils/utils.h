#ifndef FASTPOINTS_UTILS_H
#define FASTPOINTS_UTILS_H

#include <iostream>

template <typename T> T next_power_of_two(T unsigned_decimal) {

    if(std::numeric_limits<T>::is_signed) {
        std::cerr << "Datatype must be unsigned!" << std::endl;
        return 0;
    }

    if(!std::numeric_limits<T>::is_integer) {
        std::cerr << "Datatype must be integral type!" << std::endl;
        return 0;
    }

    size_t bit_size = sizeof(unsigned_decimal) + 3;

    unsigned_decimal--;
    for (size_t n = 1; n < bit_size; n <<= 1) {
        unsigned_decimal |= unsigned_decimal >> n;
    }
    unsigned_decimal++;

    return unsigned_decimal + (unsigned_decimal == 0);
}

#endif //FASTPOINTS_UTILS_H
