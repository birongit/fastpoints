#include <iostream>

#include "../src/point_cloud.h"
#include "../src/transform.h"
#include "../src/reader.h"
#include "../src/writer.h"

std::string ParseArguments(std::string key, int argc, char * argv[]) {

  auto pos = std::find(argv, argv+argc, key);

  return (++pos < argv+argc) ? *pos : std::string();

}

int main(int argc, char * argv[])
{

  std::string in  = ParseArguments(std::string("-i"), argc, argv);
  std::string out = ParseArguments(std::string("-o"), argc, argv);

  if ( in.empty() || out.empty() ){
    std::cout << "Usage: " << argv[0] << " -i <path/to/input.pcd> -o <path/to/output.pcd>" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read points from file
  PointCloud cloud;
  Read(in, cloud);
  std::cout << "Loaded " << cloud.points.size() << " points from file." << std::endl;

  // measure time
  std::clock_t start;
  double duration;
  start = std::clock();

  //shift and rotate points on gpu
  std::vector<double> shift {20.0, 20.0, 20.0};
  Quaternion quaternion(0.7071, 0.0, 0.0, 0.7071);

  int N = 1;

  for (int i = 0; i < N; i++) {
    cloud = ShiftPoints(cloud, shift);
    cloud = RotatePoints(cloud, quaternion);
  }

  duration = (std::clock() - start) / (double) CLOCKS_PER_SEC / N;
  std::cout << "Execution time: "<< duration << "s" << std::endl;

  Write(out, cloud);

  return 0;
}

