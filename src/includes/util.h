#ifndef UTIL
#define UTIL
#include <vector>
#include <queue>
struct cluster_distance {

  int cluster;
  double distance;
  ~cluster_distance();
};

// custom comparator to compare cluster_distances
struct Comp {

  bool operator()(const cluster_distance *a, const cluster_distance *b);
};

// wrapper around a priority queue for a Min heap
class minHeap {

private:
  std::priority_queue<cluster_distance *, std::vector<cluster_distance *>, Comp>
      distances;

public:
  ~minHeap();
  void push(cluster_distance *pair);
  int pop();
  void clear();
};

#endif
