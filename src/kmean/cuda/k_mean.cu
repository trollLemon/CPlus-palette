#include "adv_color.h"
#include "cuda_cluster.h"
#include "k_mean_cuda.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_set>

struct CUDA_COLOR_DATA {
  int r;
  int g;
  int b;
};

__device__ double euclideanDistance(CUDA_COLOR_DATA x, CUDA_COLOR_DATA y) {
  double dl = static_cast<double>(y.r) - static_cast<double>(x.r);
  double da = static_cast<double>(y.g) - static_cast<double>(x.g);
  double db = static_cast<double>(y.b) - static_cast<double>(x.b);
  return sqrt(dl * dl + da * da + db * db);
}

__global__ void recalcClusters(CUDA_COLOR_DATA *d_clusters,
                               CUDA_COLOR_DATA *d_colors, int *assignments,
                               int color_count, int *clusterStats) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < color_count) {

    int clusterId = assignments[idx];
    int numPoints = clusterStats[clusterId];

    atomicAdd(&(d_clusters[clusterId].r), d_colors[idx].r);
    atomicAdd(&(d_clusters[clusterId].g), d_colors[idx].g);
    atomicAdd(&(d_clusters[clusterId].b), d_colors[idx].b);
    __syncthreads();

    if (threadIdx.x == 0) {
      d_clusters[clusterId].r /= numPoints;
      d_clusters[clusterId].g /= numPoints;
      d_clusters[clusterId].b /= numPoints;
    }
  }
}

__global__ void assignPoints(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int *assignments, int k,
                             int color_count, int *cluster_stats) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < color_count) {

    double distance = INFINITY;
    int cluster_idx = -1;

    for (int i = 0; i < k; ++i) {

      double dist = euclideanDistance(d_clusters[i], d_colors[idx]);
      if (dist < distance) {
        cluster_idx = i;
        distance = dist;
      }
    }

    assignments[idx] = cluster_idx;
    atomicAdd(&cluster_stats[cluster_idx], 1);
  }
}

__global__ void initClusters(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int size, int seed) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    int point_idx = curand(&state) % (size - 1);
    d_clusters[idx] = d_colors[point_idx];
  }
}

std::vector<std::string>
CudaKmeanWrapper(cimg_library::CImg<unsigned char> *image, int size) {
  int height{image->height()};
  int width{image->width()};

  std::unordered_set<std::string> seen;
  CUDA_COLOR_DATA colors[height * width];
  ADV_Color *base = new ADV_Color(0, 0, 0);
  int totalPixels = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);

      base->setRGB(r, g, b);
      std::string hex = base->asHex();

      if (seen.count(hex) == 0) {
        CUDA_COLOR_DATA color;
        color.r = r;
        color.g = g;
        color.b = b;
        colors[totalPixels] = color;
        seen.insert(hex);
        totalPixels++;
      }
    }
  }

  CUDA_COLOR_DATA *d_colors;
  CUDA_COLOR_DATA *d_clusters;
  int *d_assignments;
  int *d_cluster_stats;

  cudaMalloc((void **)&d_colors, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_assignments, totalPixels * sizeof(int));
  cudaMalloc((void **)&d_cluster_stats, totalPixels * sizeof(int));
  cudaMalloc((void **)&d_clusters, size * sizeof(CUDA_COLOR_DATA));
  cudaMemset(d_assignments, -1,
             totalPixels *
                 sizeof(int)); // points are not given a cluster yet (id is -1)
 
  cudaMemset(d_cluster_stats, 1, size * sizeof(int));
  cudaMemcpy(d_colors, colors, totalPixels * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  bool loop = true;
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  std::random_device rd;
  int seed = rd();
  initClusters<<<blocksPerGrid, threadsPerBlock>>>(d_clusters, d_colors, size,
                                                   seed);
  cudaDeviceSynchronize();
 int x =0;
  while(x++!=100){

  blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
  assignPoints<<<blocksPerGrid, threadsPerBlock>>>(
      d_clusters, d_colors, d_assignments, size, totalPixels, d_cluster_stats);
  cudaDeviceSynchronize();
  recalcClusters<<<blocksPerGrid, threadsPerBlock>>>(
      d_clusters, d_colors, d_assignments, totalPixels, d_cluster_stats);


  cudaMemset(d_cluster_stats, 1, size * sizeof(int));

  }
  //}

  CUDA_COLOR_DATA *h_colors =
      (CUDA_COLOR_DATA *)malloc(size * sizeof(CUDA_COLOR_DATA));

  cudaMemcpy(h_colors, d_clusters, size * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyDeviceToHost);

  ADV_Color color_helper(0,0,0);
	
  std::vector<std::string> palette;
  for(int i = 0; i< size; ++i){
	
	  CUDA_COLOR_DATA color = h_colors[i];

	  color_helper.setRGB(color.r,color.g,color.b);
 	  palette.push_back(color_helper.asHex());
  }
  
  

  return palette;
}
