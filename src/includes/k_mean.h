/* *
 * k_mean.h
 *
 * Header for an implementation of the K_Means clustering algorithm
 * for color palette generation.
 *
 * This header includes definitions for funcitons to perform the algorithm,
 * along with wrappers and custom comparators to help with the execution
 *
 *
 *
 * */

#ifndef KMEAN
#define KMEAN
#include "color.h"
#include "cluster.h"
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

// custom comparator to sort clusters
struct ColorSort {

  bool operator()(const Cluster *a, const Cluster *b);
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


  /* *
   * Returns the Euclidean Distance between two colors   
   * */
  double EuclideanDistance(Color *a, Color *b);


  /* *
   * Performs K means clustering for palette generation:
   * returns a vector of strings for each color in the palette.
   * They are represented as hex color codes (#FFFFFF) for ease of use
   * in config files, CSS, etc...
   * */
  std::vector<std::string> KMeans(std::vector<Color *> &colors, int k);

#endif
