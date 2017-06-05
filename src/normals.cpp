#include "normals.h"

PointCloud<PointXYZINormal> Normals::estimate() {

    PointCloud<PointXYZINormal> normals;

    for (auto point : cloud.points) {

        auto neighbors = find_neighbors_naive(point, 10);
        auto covar = covariance(neighbors);

        double eig_val[3];
        double eig_vec[9];

        eigen3(&covar[0], eig_val, eig_vec);

        normals.points.push_back(PointXYZINormal(point.x, point.y, point.z, 1.0, eig_vec[6], eig_vec[7], eig_vec[8]));
    }

    return normals;

}

PointCloud<Point3D> Normals::find_neighbors_naive(Point3D &query_point, int k_neighbors) {

    double dist_thresh = DBL_MAX;
    double d;

    auto cmp = [&](Point3D left, Point3D right) { return (query_point.distance2(left)) > (query_point.distance2(right));};
    std::priority_queue<Point3D, std::vector<Point3D>, decltype(cmp)> queue(cmp);

    for (auto point : cloud.points) {
        d = query_point.distance2(point);
        if (d < dist_thresh) {
            queue.push(point);
            // TODO reduce threshold size
        }
    }

    PointCloud<Point3D> neighbors;
    for (int i = 0; i < k_neighbors; ++i) {
        Point3D p = queue.top();
        queue.pop();
        neighbors.points.push_back(p);
    }

    return neighbors;
}
