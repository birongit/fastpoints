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

    file << "FIELDS " <<  cloud.points[0].FieldsString() << std::endl;

    file << "SIZE " <<  cloud.points[0].SizeString() << std::endl;

    file << "TYPE " <<  cloud.points[0].TypeString() << std::endl;

    file << "WIDTH " << cloud.points.size() << std::endl;

    file << "HEIGHT " << "1" << std::endl;

    file << "POINTS " << cloud.points.size() << std::endl;

    file << "DATA " << "ascii" << std::endl;

}

void PrintPoints(std::ostream &file, const PointCloud &cloud) {

    for (auto point: cloud.points) {

        file << point << std::endl;

    }
}