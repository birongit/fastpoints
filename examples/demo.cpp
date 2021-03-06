#include <iostream>
#include <ctime>

#include "../src/common/point_cloud.h"
#include "../src/geometry/transform.h"
#include "../src/io/reader.h"
#include "../src/io/writer.h"
#include "../src/geometry/geometry.h"
#include "../src/utils/cuda_utils.h"
#include "../src/geometry/normals.h"
#include "../src/core/kernels.h"

std::string parse_arguments(std::string key, int argc, char * argv[]) {

  auto pos = std::find(argv, argv+argc, key);

  return (++pos < argv+argc) ? *pos : std::string();

}

int main(int argc, char * argv[])
{
  std::cout << "Starting example program using CUDA VERSION " << float(get_cuda_version()) / 1000 << std::endl;

  std::string in  = parse_arguments(std::string("-i"), argc, argv);
  std::string out = parse_arguments(std::string("-o"), argc, argv);

  if ( in.empty() || out.empty() ){
    std::cout << "Usage: " << argv[0] << " -i <path/to/input.pcd> -o <path/to/output.pcd>" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read points from file
  PointCloud<PointXYZI> cloud;
  read(in, cloud);
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
    cloud = shift_points(cloud, shift);
    cloud = rotate_points(cloud, quaternion);
  }

  // test: call functions from host
  auto mean_point = mean(cloud);
  std::cout << "Mean: " << mean_point.x << " " << mean_point.y << " " << mean_point.z << std::endl;
  auto covar = covariance(cloud);
  std::cout << "Covariance:" << std::endl;
  std::cout << covar[0] << " " << covar[1] << " " << covar[2] << std::endl;
  std::cout << covar[3] << " " << covar[4] << " " << covar[5] << std::endl;
  std::cout << covar[6] << " " << covar[7] << " " << covar[8] << std::endl;

  double eig_val[3];
  double eig_vec[9];

  eigen3(&covar[0], eig_val, eig_vec);
  std::cout << "Eigenvalues: " << eig_val[0] << " " << eig_val[1] << " " << eig_val[2] << std::endl;
  std::cout << "Eigenvectors: " << std::endl;
  std::cout << eig_vec[0] << " " << eig_vec[1] << " " << eig_vec[2] << std::endl;
  std::cout << eig_vec[3] << " " << eig_vec[4] << " " << eig_vec[5] << std::endl;
  std::cout << eig_vec[6] << " " << eig_vec[7] << " " << eig_vec[8] << std::endl;

  PointCloud<Point3D> cloud2;

  for (auto point : cloud.points) {
    cloud2.points.push_back(point);
  }

  Normals<PointXYZI> ne(cloud);
  auto normals = ne.estimate();

  auto range = reduce_max(cloud2) - reduce_min(cloud2);

  std::cout << "Range of points: x = " << range.x << "  y = " << range.y << "  z = " << range.z << std::endl;

  duration = (std::clock() - start) / (double) CLOCKS_PER_SEC / N;
  std::cout << "Execution time: "<< duration << "s" << std::endl;

  write(out, normals);

  return 0;
}

