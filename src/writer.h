#ifndef PROJECTFASTPOINTS_WRITER_H
#define PROJECTFASTPOINTS_WRITER_H

#include "point_cloud.h"
#include <fstream>
#include <iostream>

void Write(std::string path, PointCloud &cloud);

void PrintHeader(std::ostream &file, const PointCloud &cloud);

void PrintPoints(std::ostream &file, const PointCloud &cloud);


#endif //PROJECTFASTPOINTS_WRITER_H
