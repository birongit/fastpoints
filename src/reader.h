#ifndef PROJECTFASTPOINTS_READER_H
#define PROJECTFASTPOINTS_READER_H

#include <string>
#include <fstream>
#include <iostream>
#include "point_cloud.h"

void read(std::string &path, PointCloud<PointXYZI> &cloud);

void tokenize_line(std::string &line, std::string &delimiter, std::vector<std::string> &tokens);

void read_points(std::ifstream &ifstream, PointCloud<PointXYZI> &cloud);


#endif //PROJECTFASTPOINTS_READER_H
