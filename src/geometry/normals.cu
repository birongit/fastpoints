#include "normals.h"

template class Normals<PointXYZI>;
template class Normals<PointXYZINormal>;

template<typename T>
PointCloud<T> Normals<T>::find_neighbors_naive(T &query_point, int k_neighbors) {


    auto cmp = [&](T left, T right) { return (query_point.distance2(left)) < (query_point.distance2(right));};

    auto cloud_sort = cloud;

    std::sort(cloud_sort.points.begin(), cloud_sort.points.end(), cmp);

    PointCloud<T> neighbors;

    for (int i = 0; i < k_neighbors; ++i) {
        neighbors.points.push_back(cloud_sort.points[i]);
    }

    return neighbors;
}
