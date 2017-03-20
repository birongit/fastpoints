#include "writer.h"

void Write(std::string path, PointCloud &cloud) {

    std::ofstream file;

    std::cout << "Writing cloud " << path << std::endl;

    file.open(path);

    if (file.fail())
    {
        std::cerr << "Opening file " << path << " failed!" << std::endl;
        return;
    }

    PrintHeader(file, cloud);

    PrintPoints(file, cloud);

}

void PrintHeader(std::ostream &file, const PointCloud &cloud) {

    file << "VERSION " << ".7" << std::endl;

    file << "FIELDS " <<  "x y z i" << std::endl;

    file << "SIZE " <<  "4 4 4 4" << std::endl;

    file << "TYPE " <<  "F F F F" << std::endl;

    file << "WIDTH " << cloud.points.size() << std::endl;

    file << "HEIGHT " << "1" << std::endl;

    file << "POINTS " << cloud.points.size() << std::endl;

    file << "DATA " << "ascii" << std::endl;

}

void PrintPoints(std::ostream &file, const PointCloud &cloud) {

    for (auto point: cloud.points) {

        file << (float) point.x << " " << (float) point.y << " " << (float) point.z << " " << (float) point.i << std::endl;

    }
}