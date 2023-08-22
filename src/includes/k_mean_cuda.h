#ifndef CUDA_LINK_INTERFACE
#define CUDA_LINK_INTERFACE
#include <vector>
#include <string>
#include "CImg.h"
//TODO: inherit stuff from main k_mean class


#include "adv_color.h"
#include "cuda_cluster.h"
#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>

class CUDAKMean {
typedef CudaCluster Cluster;

private:
  ADV_Color *centroid;
  std::vector<ADV_Color *> colors;
  std::unordered_map<int /*Cluster ID*/, Cluster *> clusters; // clusters


  int id;
  /* *
   * inits the process for K_Means.
   *
   * K points are selected to create our initial clusters.
   *
   * */
  void K_MEAN_INIT(int k);

  /* *
   * Performs the Clustering.
   * Once the method is finished, the remaining centroids will
   * be our color palette
   *
   * */
  void K_MEAN_START();

  /* *
   * Returns the Euclidian Distance between two colors by looking
   * at the colors' LAB values
   * Assumes the Colors had their RGB values converted to LAB values
   * */
  double EuclidianDistance(ADV_Color *a, ADV_Color *b);

  /* *
   * Recalculates centroids for all clusters in the given set
   * This will be called in K_MEAN_START every iteration if Clusters gain or
   * lose points.
   * */
  void reCalculateCentroids(std::set<Cluster *>);

public:
  /* *
   * returns a vector of strings for each color in the palette.
   * They are represented as hex color codes (#FFFFFF) for ease of use
   * in config files, CSS, etc...
   * */
  std::vector<std::string> makePalette(std::vector<ADV_Color *> &colors, int k);
};



std::vector<std::string> CudaKmeanWrapper(cimg_library::CImg<unsigned char> *image, int size);


#endif
