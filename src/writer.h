#ifndef PROJECTFASTPOINTS_WRITER_H
#define PROJECTFASTPOINTS_WRITER_H

#include "point_cloud.h"
#include <fstream>
#include <iostream>

template <typename T> void print_header(std::ostream &file, const PointCloud<T> &cloud) {

    file << "VERSION " << ".7" << std::endl;

    file << "FIELDS " <<  cloud.points[0].FieldsString() << std::endl;

    file << "SIZE " <<  cloud.points[0].SizeString() << std::endl;

    file << "TYPE " <<  cloud.points[0].TypeString() << std::endl;

    file << "WIDTH " << cloud.points.size() << std::endl;

    file << "HEIGHT " << "1" << std::endl;

    file << "POINTS " << cloud.points.size() << std::endl;

    file << "DATA " << "ascii" << std::endl;

}

template <typename T> void print_points(std::ostream &file, const PointCloud<T> &cloud) {

    for (auto point: cloud.points) {

        file << point << std::endl;

    }
}

template <typename T> void write(std::string path, PointCloud<T> &cloud) {

    std::ofstream file;

    std::cout << "Writing cloud " << path << std::endl;

    file.open(path);

    if (file.fail())
    {
        std::cerr << "Opening file " << path << " failed!" << std::endl;
        return;
    }

    print_header(file, cloud);

    print_points(file, cloud);

}


#endif //PROJECTFASTPOINTS_WRITER_H
