#ifndef PROJECTFASTPOINTS_READER_H
#define PROJECTFASTPOINTS_READER_H

#include <string>
#include <fstream>
#include <iostream>
#include "point_cloud.h"

void Read(std::string &path, PointCloud &cloud);

void TokenizeLine(std::string& line, std::string& delimiter, std::vector<std::string>& tokens);

void ReadPoints(std::ifstream &ifstream, PointCloud &cloud);


#endif //PROJECTFASTPOINTS_READER_H