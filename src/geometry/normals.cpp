#include "normals.h"

PointCloud<PointXYZINormal> Normals::estimate() {

    PointCloud<PointXYZINormal> normals;
    double eig_val[3], eig_vec[9];

    for (auto point : cloud.points) {

        auto neighbors = find_neighbors_naive(point, 10);
        auto covar = covariance(neighbors);

        eigen3(&covar[0], eig_val, eig_vec);

        normals.points.push_back(PointXYZINormal(point.x, point.y, point.z, 1.0, eig_vec[6], eig_vec[7], eig_vec[8]));
    }

    return normals;

}

PointCloud<Point3D> Normals::find_neighbors_naive(Point3D &query_point, int k_neighbors) {


    auto cmp = [&](Point3D left, Point3D right) { return (query_point.distance2(left)) < (query_point.distance2(right));};

    auto cloud_sort = cloud;

    std::sort(cloud_sort.points.begin(), cloud_sort.points.end(), cmp);

    PointCloud<Point3D> neighbors;

    for (int i = 0; i < k_neighbors; ++i) {
        neighbors.points.push_back(cloud_sort.points[i]);
    }

    return neighbors;
}
