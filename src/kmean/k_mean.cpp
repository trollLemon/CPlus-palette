#include "k_mean.h"
#include "cluster.h"
#include "color.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#define MAX_IRERATIONS 12
cluster_distance::~cluster_distance() {}
void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
}

bool ColorSort::operator()(const Cluster *a, const Cluster *b) {

  return a->getCentroid()->Red() > b->getCentroid()->Red() &&
         a->getCentroid()->Green() > b->getCentroid()->Green() &&
         a->getCentroid()->Blue() > b->getCentroid()->Blue();
}

bool Comp::operator()(const cluster_distance *a, const cluster_distance *b) {
  return a->distance < b->distance;
}

minHeap::~minHeap() { clear(); }

int minHeap::pop() {

  if (distances.top() == nullptr) {
    return -1;
  }

  int ClusterId = distances.top()->cluster;
  delete distances.top();
  distances.pop();

  return ClusterId;
}
void minHeap::clear() {

  while (!distances.empty()) {
    delete distances.top();
    distances.pop();
  }
}

double EuclidianDistance(Color *a, Color *b) {

  double deltaR = a->Red() - b->Red();
  double deltaG = a->Green() - b->Green();
  double deltaB = a->Blue() - b->Blue();

  return ((deltaR * deltaR) + (deltaG * deltaG) + (deltaB * deltaB));
}

std::vector<std::string> KMeans(std::vector<Color *> &colors, int k) {

  std::map<Color *, minHeap *> data; // Distances from points to each centroid
  std::unordered_map<int /*Cluster ID*/, Cluster *> clusters; // clusters

  // load our data into a Red Black tree, in this case a normal map
  // While this data will not be used by the quantizer for the first iteration,
  // the data will every iteration after the first iteration.
  for (Color *point : colors) {
    data[point] = new minHeap();
  }

  /*
   * Initialize the Clustering
   *
   * */
  // randomly get k points
  int size = colors.size();

  // better seeding from
  // https://www.learncpp.com/cpp-tutorial/generating-random-numbers-using-mersenne-twister/
  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{size / 2, size};
  std::set<int> seen; // make sure we have unique numbers
  std::vector<int> colorIdx;
  while (seen.size() != static_cast<long unsigned int>(k)) {
    int num = kPoints(mt);
    if (seen.count(num) == 0) {
      seen.insert(num);
      colorIdx.push_back(num);
    }
  }

  for (int i = 0; i < k; ++i) {
    Cluster *cluster = new Cluster(colors[colorIdx[i]], i);
    clusters[i] = cluster;
  }

  // Do the first iteration
  for (Color *point : colors) {
    point->setClusterId(-1);
  }

  // first phase done

  /*
   * Start the clustering iteration
   *
   * */
  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,

  std::set<Cluster *> toRecalculate;
  int iterations = 0;

  do {
    toRecalculate.clear();
    for (Color *point : colors) {

      minHeap *heap = data[point];

      for (auto cluster : clusters) {

        double distance =
            EuclidianDistance(cluster.second->getCentroid(), point);
        int id = cluster.second->getId();
        cluster_distance *dist = new cluster_distance;
        dist->cluster = id;
        dist->distance = distance;
        heap->push(dist);
      }

      int id = heap->pop();

      if (id != point->getClusterId()) {
        toRecalculate.insert(clusters[id]);
        int pId = point->getClusterId();
        if (pId != -1 ) {
          toRecalculate.insert(clusters[pId]);
        }
      }

      point->setClusterId(id);
    }
    for (Cluster *cluster : toRecalculate) {
      cluster->calcNewCentroid();
    }

  } while (toRecalculate.size() != 0 && iterations++ < MAX_IRERATIONS);

  // At this point the clusters have converges, so we can collect the color
  // palette

  std::vector<std::string> palette;
  std::vector<Cluster *> sortedColors;

  for (auto cluster : clusters) {
    // up until this point the Colors have had their LAB Values used, So we
    // should update the RGB values
    sortedColors.push_back(cluster.second);
  }

  std::sort(sortedColors.begin(), sortedColors.end(), ColorSort());

  for (Cluster *cluster : sortedColors) {
    // up unil this point the Colors have had their LAB Values used, So we
    // should update the RGB values
    cluster->getCentroid();
    palette.push_back(cluster->asHex());
  }

  for (auto heap : data) {
    delete heap.second;
  }

  for (auto cluster : clusters) {
    delete cluster.second;
  }

  return palette;
}
