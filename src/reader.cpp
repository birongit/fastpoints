#include <thread>
#include "reader.h"

void read(std::string &path, PointCloud<PointXYZI> &cloud) {

    std::string line;
    std::string delimiter = " ";
    std::vector<std::string> elements;

    std::cout << "Reading cloud " << path << std::endl;

    std::ifstream file {path};

    if (!file.good()) {
        std::cerr << "File not found! Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    do {
        getline(file, line);
        tokenize_line(line, delimiter, elements);
    } while ((elements[0] != std::string("DATA")));

    if (elements[1] != std::string("ascii")) {
        std::cerr << "File format \"" << elements[1] << "\" currently not supported! Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    read_points(file, cloud);

}

void read_points(std::ifstream &ifstream, PointCloud<PointXYZI> &cloud) {

    while (!ifstream.eof()) {
        double x, y, z;
        double i = 1.0;
        ifstream >> x >> y >> z;
        cloud.points.push_back(PointXYZI(x, y, z, i));
    }
}

void tokenize_line(std::string &line, std::string &delimiter, std::vector<std::string> &tokens) {

    tokens.clear();

    while(line.find(delimiter) != std::string::npos) {
        auto pos = line.find(delimiter);
        tokens.push_back(line.substr(0, pos));
        line = line.substr(++pos, line.size());
    };
    tokens.push_back(line);

}
