#include "k_mean.h"
#include "cluster.h"
#include "color.h"
#include <algorithm>
#include <random>
#define MAX_ITERATIONS 12
cluster_distance::~cluster_distance() {}
void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
}

bool ColorSort::operator()(const Color *a, const Color *b) {
  return a->Red() > b->Red() && a->Green() > b->Green() &&
         a->Blue() > b->Blue();
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

double EuclideanDistance(Color *a, Color *b) {

  double deltaR = a->Red() - b->Red();
  double deltaG = a->Green() - b->Green();
  double deltaB = a->Blue() - b->Blue();

  return ((deltaR * deltaR) + (deltaG * deltaG) + (deltaB * deltaB));
}

std::vector<Color *> KMeans(std::vector<Color *> &colors, int k) {

  std::map<Color *, minHeap *> data; // Distances from points to each centroid
  std::unordered_map<int /*Cluster ID*/, Cluster *> clusters; // clusters

  for (Color *point : colors) {
    data[point] = new minHeap();
  }

  /*
   * Initialize the Clustering
   *
   * */
  // randomly get k points
  int size = colors.size();

  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{size / 2, size};
  std::set<int> seen; // make sure we have unique numbers
  std::vector<int> colorIdx;
  while (seen.size() != static_cast<long unsigned int>(k)) {
    int num = kPoints(mt);

    if (seen.count(num) != 0)
      continue;

    seen.insert(num);

    colorIdx.push_back(num);
  }

  for (int i = 0; i < k; ++i)
    clusters[i] = new Cluster(colors[colorIdx[i]], i);

  // Do the first iteration
  for (Color *point : colors)
    point->setClusterId(-1);

  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,

  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,

  std::set<Cluster *> toRecalculate;

  int itrs = 0;
  do {
    toRecalculate.clear();
    for (Color *point : colors) {

      minHeap *heap = data[point];

      for (auto cluster : clusters) {

        double distance =
            EuclideanDistance(cluster.second->getCentroid(), point);
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
	
    for (Cluster *cluster : toRecalculate)
      cluster->calcNewCentroid();

    itrs++;
  } while (itrs < MAX_ITERATIONS && toRecalculate.size() !=0);

  std::vector<std::string> palette;
  std::vector<Color *> sortedColors;

  for (auto cluster : clusters) {
    sortedColors.push_back(cluster.second->getCentroid());
    delete cluster.second;
  }
  std::sort(sortedColors.begin(), sortedColors.end(), ColorSort());

  for (auto heap : data)
    delete heap.second;

  return sortedColors;
}
