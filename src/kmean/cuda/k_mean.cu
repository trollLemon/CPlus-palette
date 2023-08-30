#include "adv_color.h"
#include "cuda_cluster.h"
#include "k_mean_cuda.h"
#include "util.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_set>
#define THREADS_PER_BLOCK 256
typedef pixel CUDA_COLOR_DATA;

__device__ double euclideanDistance(CUDA_COLOR_DATA x, CUDA_COLOR_DATA y) {
  double dl = static_cast<double>(y.r) - static_cast<double>(x.r);
  double da = static_cast<double>(y.g) - static_cast<double>(x.g);
  double db = static_cast<double>(y.b) - static_cast<double>(x.b);
  return sqrt(dl * dl + da * da + db * db);
}

__global__ void sumClusters(CUDA_COLOR_DATA *d_centroids,
                            CUDA_COLOR_DATA *d_colors, int *assignments,
                            int *d_clusterSizes, CUDA_COLOR_DATA *partialSums,
                            int k, int color_count) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (id > color_count)
    return;

  __shared__ CUDA_COLOR_DATA s_datapoints[THREADS_PER_BLOCK];
  s_datapoints[tid] = d_colors[id];

  __shared__ int s_clust_assn[THREADS_PER_BLOCK];
  s_clust_assn[tid] = assignments[id];

  /* int clusterId = assignments[id];
  int r, g, b;

  r = d_colors[id].r;
  g = d_colors[id].g;
  b = d_colors[id].b;

  atomicAdd(&d_clusterSizes[clusterId], 1);
  atomicAdd(&d_centroids[clusterId].r, r);
  atomicAdd(&d_centroids[clusterId].g, g);
  atomicAdd(&d_centroids[clusterId].b, b);
*/

  __syncthreads();

  if (tid == 0) {

    CUDA_COLOR_DATA b_clust_datapoint_sums[THREADS_PER_BLOCK] = {0};
    for (int j = 0; j < blockDim.x; ++j) {
      int clusterId = s_clust_assn[j];
      int r, g, b;

      r = s_datapoints[id].r;
      g = s_datapoints[id].g;
      b = s_datapoints[id].b;

      b_clust_datapoint_sums[clusterId].r += r;
      b_clust_datapoint_sums[clusterId].g += g;
      b_clust_datapoint_sums[clusterId].b += b;
    }

    for (int z = 0; z < k; ++z) {

      atomicAdd(&d_centroids[z].r, b_clust_datapoint_sums[z].r);
      atomicAdd(&d_centroids[z].g, b_clust_datapoint_sums[z].g);
      atomicAdd(&d_centroids[z].b, b_clust_datapoint_sums[z].b);
    }
    __syncthreads();
  }
}

__global__ void recalcClusters(CUDA_COLOR_DATA *d_centroids, int *d_clust_sizes,
                               int k) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > k)
    return;

  int size = d_clust_sizes[idx];

  d_centroids[idx].r /= size;
  d_centroids[idx].g /= size;
  d_centroids[idx].b /= size;
}

__global__ void assignPoints(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int *assignments,
                             int *cluster_counts, int k, int color_count) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > color_count)
    return;

  double min_dist = INFINITY;
  int closest_centroid = 0;

  for (int i = 0; i < k; ++i) {
    double dist = euclideanDistance(d_clusters[i], d_colors[idx]);

    if (dist < min_dist) {
      min_dist = dist;
      closest_centroid = i;
    }
  }
  assignments[idx] = closest_centroid;
}

__global__ void initClusters(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int size,
                             int *cluster_picks) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    d_clusters[idx] = d_colors[cluster_picks[idx]];
  }
}

std::vector<std::string> CudaKmeanWrapper(CUDA_COLOR_DATA *pixel_data, int size,
                                          int totalPixels) {

  CUDA_COLOR_DATA *d_colors;
  CUDA_COLOR_DATA *d_clusters;
  CUDA_COLOR_DATA *d_partialSums;

  int *d_assignments;
  int *d_random_points;
  int *d_clust_sizes;

  cudaMalloc((void **)&d_colors, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_partialSums, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_random_points, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_assignments, totalPixels * sizeof(int));
  cudaMalloc((void **)&d_clusters, size * sizeof(CUDA_COLOR_DATA));
  cudaMalloc(&d_clust_sizes, size * sizeof(int));
  cudaMemcpy(d_colors, pixel_data, totalPixels * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{totalPixels / 2, totalPixels};
  std::set<int> seen; // make sure we have unique numbers
  int *colorIndecies = new int[size];
  int colorIndecies_idx = 0;
  while (colorIndecies_idx != size) {
    int num = kPoints(mt);
    if (seen.count(num) == 0) {
      seen.insert(num);
      colorIndecies[colorIndecies_idx] = num;
      colorIndecies_idx++;
    }
  }

  cudaMemcpy(d_random_points, colorIndecies,
             colorIndecies_idx * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  initClusters<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_clusters, d_colors, size,
                                                     d_random_points);

  cudaDeviceSynchronize();

  blocksPerGrid = (totalPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int x = 0;
  while (x != 10) {

    assignPoints<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_clusters, d_colors, d_assignments, d_clust_sizes, size, totalPixels);

    sumClusters<<<blocksPerGrid, THREADS_PER_BLOCK,
                  size * sizeof(CUDA_COLOR_DATA)>>>(
        d_clusters, d_colors, d_assignments, d_clust_sizes, d_partialSums, size,
        totalPixels);

    recalcClusters<<<size, 1>>>(d_clusters, d_clust_sizes, size);
    cudaMemset(d_clust_sizes, 0, size * sizeof(int));
    cudaDeviceSynchronize();
    x++;
  }

  CUDA_COLOR_DATA *h_colors =
      (CUDA_COLOR_DATA *)malloc(size * sizeof(CUDA_COLOR_DATA));

  cudaMemcpy(h_colors, d_clusters, size * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyDeviceToHost);

  cudaFree(d_colors);
  cudaFree(d_random_points);
  cudaFree(d_assignments);
  cudaFree(d_clusters);

  ADV_Color color_helper(0, 0, 0);
  std::vector<std::string> palette;
  for (int i = 0; i < size; ++i) {

    CUDA_COLOR_DATA color = h_colors[i];

    color_helper.setRGB(color.r, color.g, color.b);

    palette.push_back(color_helper.asHex());
  }

  free(h_colors);

  return palette;
}
