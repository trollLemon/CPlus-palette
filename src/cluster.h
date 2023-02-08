#ifndef CLUSTER
#define CLUSTER
#include "color.h"
#include <vector>
#include <unordered_map>
class Cluster {

private:
  Color *centroid;
  std::unordered_map<int,Color *> points;

public:
  Cluster();
  ~Cluster();
  void calcNewCentroid();
  std::string asHex();
  void addPoint();
};

#endif
