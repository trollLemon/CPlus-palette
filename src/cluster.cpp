#include "cluster.h"
#include "color.h"

Color *Cluster::getCentroid() const { return centroid; }
Cluster::Cluster(Color *a, int id) : centroid{a}, id{id} {}
Cluster::~Cluster() {}

void Cluster::calcNewCentroid() {

  double sumL = 0;
  double sumA = 0;
  double sumB = 0;
  double size = (double)points.size();
  for (auto color : points) {

    sumL += color.second->Lum();
    sumA += color.second->aVal();
    sumB += color.second->bVal();
  }
  if (size == 0) {
    return;
  }

  // modify this clusters centroid values with the new averages,
  // this is better than reallocating memory for a new point each time
  centroid->setLAB(sumL / size, sumA / size, sumB / size);
}

std::string Cluster::asHex() { return centroid->asHex(); }
int Cluster::getId() { return id; }
void Cluster::addPoint(int pId, Color *point) { points[pId] = point; }
