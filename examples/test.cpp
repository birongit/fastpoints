#include <iostream>

#include "../src/point_cloud.h"
#include "../src/transform.h"

int main(int argc, char * argv[])
{
  PointCloud cloud;

  // generate some points for testing
  int x,y,z;

  for (int j = 0; j < 1000; ++j) {
    x = j;
    y = - j;
    z = 0;
    cloud.points.push_back(PointXYZI(x,y,z,1));
  }

  std::vector<double> shift {20.0, 20.0, 20.0};

  cloud = shiftPoints(cloud, shift);

  return 0;
}

