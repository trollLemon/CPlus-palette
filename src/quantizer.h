#ifndef COLOR_QUANT
#define COLOR_QUANT
#include "cluster.h"
#include "color.h"
#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>

// Cluster id and the distance from a point to its centroid
struct cluster_distance {

  int cluster;
  double distance;
  ~cluster_distance();
};



struct ColorSort {

  bool operator()(const Cluster *a, const Cluster *b);
};

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

// color quantizer
class Quantizer {

private:
  std::map<Color *, minHeap *> data; // Distances from points to each centroid
  std::unordered_map<int /*Cluster ID*/, Cluster *> clusters; // clusters
  void K_MEAN_INIT(int k); // select k random points to be the initial Clusters
  void K_MEAN_START();     // begins the data clustering
  double EuclidianDistance(Color *a, Color *b);
  std::vector<Color *> colors;
  void reCalculateCentroids(std::set<Cluster *>);

public:
  // returns a vector of strings for each color in the palette.
  // They are represented as hex color codes (#FFFFFF) for ease of use
  // in config files, CSS, etc...
  std::vector<std::string> makePalette(std::vector<Color *> &colors, int k);
};

#endif
