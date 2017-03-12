#include <iostream>

#include "../src/point_cloud.h"
#include "../src/transform.h"
#include "../src/reader.h"

int main(int argc, char * argv[])
{

  std::string filepath;

  auto pos = std::find(argv, argv+argc, std::string("-f"));

  if (++pos < argv+argc) {
    filepath = *pos;
    std::cout << "Reading cloud " << filepath << std::endl;
  } else {
    std::cout << "Usage: " << argv[0] << " -f <path/to/cloud>" << std::endl;
  }

  // read points from file
  PointCloud cloud;
  Read(filepath, cloud);
  std::cout << "Loaded " << cloud.points.size() << " points from file." << std::endl;

  // measure time
  std::clock_t start;
  double duration;
  start = std::clock();

  //shift points on gpu
  std::vector<double> shift {20.0, 20.0, 20.0};
  cloud = ShiftPoints(cloud, shift);

  duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
  std::cout << "Execution time: "<< duration << "s" << std::endl;

  return 0;
}

