#include "quantizer.h"
#include "cluster.h"
#include "color.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <random>

void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
}

bool Comp::operator()(const cluster_distance *a, const cluster_distance *b) {
  return a->distance < b->distance;
}

int minHeap::pop() {

  int ClusterId = distances.top()->cluster;
  distances.pop();

  return ClusterId;
}
void minHeap::clear() {

  while (!distances.empty()) {
    delete distances.top();
    distances.pop();
  }
}

double Quantizer::EuclidianDistance(Color *a, Color *b) {

  double deltaL = a->Lum() - b->Lum();
  double deltaA = a->aVal() - b->aVal();
  double deltaB = a->bVal() - b->bVal();

  return std::sqrt((deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB));
}

void Quantizer::K_MEAN_INIT(int k) {

  // randomly get k points
  auto randomGen = std::default_random_engine{};

  // TODO: gen random numbers instead of shuffling the entire array
  //  shuffle colors array
  //  we only need to do this once
  std::shuffle(colors.begin(), colors.end(), randomGen);

  for (int i = 0; i < k; ++i) {
    Cluster *cluster = new Cluster(colors[i], i);
    clusters[i] = cluster;
  }

  // Do the intial iteration
  for (Color *point : colors) {

    minHeap *heap = data[point];
    for (auto Cluster : clusters) {

      double distance = EuclidianDistance(Cluster.second->getCentroid(), point);
      int id = Cluster.second->getId();

      cluster_distance *clusterDist = new cluster_distance;
      clusterDist->cluster = id;
      clusterDist->distance = distance;
      heap->push(clusterDist);
    }

    int closestCluster = heap->pop();
    clusters[closestCluster]->addPoint(point->getId(), point);

    point->setClusterId(closestCluster);
  }

  for (auto Cluster : clusters) {
    Cluster.second->calcNewCentroid();
  }
  // first phase done
  // the next iterations will attempt to only recalculate centroids and
  // distances for the centroid that have points added or removed.
}
void Quantizer::K_MEAN_START() {

  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,
  // so K_MEAN_START will be able to recalc their centroids

  std::set<Cluster *> toRecalculate;

  
  bool firstRun = true;
  while (firstRun || toRecalculate.size()!=0) {
    
    toRecalculate.clear();
      for (Color *point : colors) {

      minHeap *heap = data[point];

      for (auto Cluster : clusters) {

        double distance =
            EuclidianDistance(Cluster.second->getCentroid(), point);
        int id = Cluster.second->getId();

        cluster_distance *clusterDist = new cluster_distance;
        clusterDist->cluster = id;
        clusterDist->distance = distance;
        heap->push(clusterDist);
      }

      int closestCluster = heap->pop();

      if (closestCluster != point->getClusterId()) {
        toRecalculate.insert(clusters[closestCluster]);
        toRecalculate.insert(clusters[point->getClusterId()]);
      }

      point->setClusterId(closestCluster);
    }
    for (auto i : toRecalculate) {
      i->calcNewCentroid();
    }
	
    firstRun= false;
  }
}
std::vector<std::string> Quantizer::makePalette(std::vector<Color *> &colors,
                                                int k) {

  this->colors = colors;

  // load our data into a Red Black tree, in this case a normal map
  // While this data will not be used by the quantizer for the first iteration,
  // the data will every iteration after the first iteration.
  for (Color *point : colors) {
    data[point] = new minHeap();
  }

  /* Initialize the Clustering
   *
   * The first iteration will  be run here aswell
   *
   * */
  K_MEAN_INIT(k);

  /**/

  K_MEAN_START();

  std::vector<std::string> palette;
  for (auto cluster : clusters) {

    // up unil this point the Colors have had their LAB Values used, So we
    // should update the RGB values
    cluster.second->getCentroid()->LABtoRGB();
    palette.push_back(cluster.second->asHex());
  }

  return palette;
}
