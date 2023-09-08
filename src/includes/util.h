#ifndef UTIL
#define UTIL
#include <vector>
#include <queue>
struct pixel {
  int r;
  int g;
  int b;
};

struct non_rgb_colorspace {

	double a;
	double b;
	double c;

};

struct adv_pixel {
  
	pixel rgb;
        non_rgb_colorspace xyz;
	non_rgb_colorspace lab;
};
  // D65/2° standard illuminants for X Y and Z
  const double X_2 = 95.047;
  const double Y_2 = 100.000;
  const double Z_2 = 108.883;



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

non_rgb_colorspace rgb_to_xyz(int r, int g, int b);
non_rgb_colorspace xyz_to_lab(double x, double y, double z);
non_rgb_colorspace lab_to_xyz(double l, double a, double b);
pixel xyz_to_rgb(double x, double y, double z);

#endif
