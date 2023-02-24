#ifndef CLUSTER
#define CLUSTER
#include "color.h"
#include <unordered_map>
#include <vector>
class Cluster {

private:
  Color *centroid;
  std::unordered_map<int, Color *> points;
  int id;

public:
  Color *getCentroid();
  Cluster(Color *a, int id);
  ~Cluster();
  void calcNewCentroid();
  int getId();
  std::string asHex();
  void addPoint(int id, Color *points);
};

#endif
