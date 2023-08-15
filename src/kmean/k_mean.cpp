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

cluster_distance::~cluster_distance() {}
void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
}

bool ColorSort::operator()(const Cluster *a, const Cluster *b) {

  return a->getCentroid()->Lum() > b->getCentroid()->Lum();
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

double KMean::EuclidianDistance(ADV_Color *a, ADV_Color *b) {

  double deltaL = a->Lum() - b->Lum();
  double deltaA = a->aVal() - b->aVal();
  double deltaB = a->bVal() - b->bVal();

  return std::sqrt((deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB));
}

void KMean::K_MEAN_INIT(int k) {

  // randomly get k points
  int size = colors.size();

  // better seeding from
  // https://www.learncpp.com/cpp-tutorial/generating-random-numbers-using-mersenne-twister/
  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{size / 2, size};
  std::set<int> seen; // make sure we have unique numbers
  std::vector<int> colorIndecies;
  while (seen.size() != static_cast<long unsigned int>(k)) {
    int num = kPoints(mt);
    if (seen.count(num) == 0) {
      seen.insert(num);
      colorIndecies.push_back(num);
    }
  }

  for (int i = 0; i < k; ++i) {
    Cluster *cluster = new Cluster(colors[colorIndecies[i]], i);
    clusters[i] = cluster;
  }

  // Do the intial iteration
  for (ADV_Color *point : colors) {
    point->setClusterId(-1);
  }

  // first phase done
}
void KMean::K_MEAN_START() {

  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,

  std::set<Cluster *> toRecalculate;

  bool loop = true;

  int oldSize = 0;
  int newSize = 0;
  int failCount = 0; // how many times there are still clusters left in the

  while (loop || toRecalculate.size() != 0) {
    if (oldSize == newSize) {
      failCount++;
    }

    // if we fail to converge the clusters more than 5 times,
    // we exit
    if (failCount > 5) {
      return;
    }

    oldSize = toRecalculate.size();
    toRecalculate.clear();
    for (ADV_Color *point : colors) {

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
        if (pId != -1) {
          toRecalculate.insert(clusters[pId]);
        }
      }

      point->setClusterId(id);
    }
    if (toRecalculate.size() == 0) {
      return;
    }
    for (Cluster *cluster : toRecalculate) {
      cluster->calcNewCentroid();
    }

    loop = !loop;
  }
}

std::vector<std::string> KMean::makePalette(std::vector<ADV_Color *> &colors,
                                            int k) {

  this->colors = colors;

  // load our data into a Red Black tree, in this case a normal map
  // While this data will not be used by the quantizer for the first iteration,
  // the data will every iteration after the first iteration.
  for (ADV_Color *point : colors) {
    data[point] = new minHeap();
  }

  /*
   * Initialize the Clustering
   *
   * */
  K_MEAN_INIT(k);

  /*
   * Start the clustring iteration
   *
   * */
  K_MEAN_START();

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
    cluster->getCentroid()->LABtoRGB();
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
