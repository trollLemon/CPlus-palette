#include "k_mean.h"
#include "cluster.h"
#include "color.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <unordered_map>
#define MAX_ITERATIONS 256

struct CompareColors {
  bool operator()(Color *a, Color *b) const { return a->asHex() > b->asHex(); }
};

double EuclideanDistance(Color *a, Color *b) {

  double deltaR = a->Red() - b->Red();
  double deltaG = a->Green() - b->Green();
  double deltaB = a->Blue() - b->Blue();

  double dist2 = ((deltaR * deltaR) + (deltaG * deltaG) + (deltaB * deltaB));

  return sqrt(dist2);
}

std::vector<Color *> KMeans(std::vector<Color *> &colors, int k) {

  std::unordered_map<int, Cluster *> clusters;

  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 gen(ss); // Seed with current time
  std::uniform_int_distribution<int> dist(0, colors.size() - 1);

  for (int i = 0; i < k; i++) {
    clusters[i] = new Cluster(colors[dist(gen)], i);
  }

  for (Color *point : colors) {
    point->setClusterId(-1);
  }

  std::set<int> toRecalculate;
  for (int i = 0; i < MAX_ITERATIONS; ++i) {

    toRecalculate.clear();
    for (Color *point : colors) {

      int id = -1;

      double maxDistance = INFINITY;
      for (std::pair<int, Cluster *> clusterPairs : clusters) {

        Cluster *cluster = clusterPairs.second;
        double distance = EuclideanDistance(cluster->getCentroid(), point);

        if (distance < maxDistance) {
          maxDistance = distance;
          id = cluster->getId();
        }
      }

      int pid = point->getClusterId();

      if (id != pid) {
        point->setClusterId(id);
        clusters[id]->addPoint(point);
        toRecalculate.insert(id);
        // edge case for when the points all have an of -1 for the first run
        if (pid > -1) {
          toRecalculate.insert(pid);
        }
      }
    }

    if (toRecalculate.empty()) {
      break; // if no points moved, we converged
    }

    for (int cluster : toRecalculate) {
      clusters[cluster]->calcNewCentroid();
    }
  }

  std::vector<Color *> sortedColors;
  for (auto cluster : clusters) {
    sortedColors.push_back(cluster.second->getCentroid());
    delete cluster.second;
  }

  std::sort(std::begin(sortedColors), std::end(sortedColors), CompareColors());

  return sortedColors;
}
