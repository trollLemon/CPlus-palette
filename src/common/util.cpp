#include "util.h"
#include "cluster.h"
cluster_distance::~cluster_distance() {}
void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
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
