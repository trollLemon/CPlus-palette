#ifndef CUDA_LINK_INTERFACE
#define CUDA_LINK_INTERFACE
#include <vector>
#include <string>
#include "CImg.h"
#include "util.h"
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


public:
  /* *
   * returns a vector of strings for each color in the palette.
   * They are represented as hex color codes (#FFFFFF) for ease of use
   * in config files, CSS, etc...
   * */
  std::vector<std::string> makePalette(std::vector<ADV_Color *> &colors, int k);
};



std::vector<std::string> CudaKmeanWrapper(pixel* colors, int size, int totalPixels);


#endif
