#include "cluster.h"
#include "color.h"

Color *Cluster::getCentroid() const { return centroid; }
Cluster::Cluster(Color *a, int id) : centroid{a}, id{id} {}
Cluster::~Cluster() {}

void Cluster::calcNewCentroid() {

  double sumR = 0;
  double sumG = 0;
  double sumB = 0;
  double size = (double)points.size();
  for (auto color : points) {

    sumR += color->Red();
    sumG += color->Green();
    sumB += color->Blue();
  }
  if (size == 0) {
    return;
  }

  // modify this clusters centroid values with the new averages,
  // this is better than reallocating memory for a new point each time
  centroid->setRGB(sumR / size, sumG / size, sumB / size);
  points.clear();
}

std::string Cluster::asHex() { return centroid->asHex(); }
int Cluster::getId() { return id; }
void Cluster::addPoint(Color *point) { points.push_back(point); }
