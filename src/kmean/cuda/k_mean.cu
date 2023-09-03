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
#define BLOCKSIZE 256
typedef pixel CUDA_COLOR_DATA;

__device__ double euclideanDistance(CUDA_COLOR_DATA x, CUDA_COLOR_DATA y) {
  double dl = static_cast<double>(y.r) - static_cast<double>(x.r);
  double da = static_cast<double>(y.g) - static_cast<double>(x.g);
  double db = static_cast<double>(y.b) - static_cast<double>(x.b);
  return sqrt(dl * dl + da * da + db * db);
}

__global__ void sumClusters(CUDA_COLOR_DATA *d_centroids,
                            CUDA_COLOR_DATA *d_colors, int *assignments,
                            int *d_clusterSizes, int* d_clust_recalcs, CUDA_COLOR_DATA *partialSums,
                            int k, int color_count) {

  extern __shared__ CUDA_COLOR_DATA shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= color_count)
    return;

  
  int assn_cluster = assignments[global_index];
  
  if(d_clust_recalcs[assn_cluster]) return;
  CUDA_COLOR_DATA color = d_colors[global_index];
  CUDA_COLOR_DATA blank = {0, 0, 0};
  for (int cluster = 0; cluster < k; ++cluster) {
    
     shared_data[local_index] = (assn_cluster == cluster) ? color : blank;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
      if (local_index < stride) {

        shared_data[local_index].r += shared_data[local_index + stride].r;
        shared_data[local_index].g += shared_data[local_index + stride].g;
        shared_data[local_index].b += shared_data[local_index + stride].b;
      }
    }
    __syncthreads();

    if (local_index == 0) {
      
      const int cluster_index = blockIdx.x * k + cluster;
	partialSums[cluster_index] = shared_data[local_index];
    }
  }
}
__global__ void recalcClusters(CUDA_COLOR_DATA *d_centroids,
                               CUDA_COLOR_DATA *partialSums, int *d_clust_sizes, int* d_clust_recalcs,
                               int k) {
  extern __shared__ CUDA_COLOR_DATA shared_data[];

  const int index = threadIdx.x;
  shared_data[index] = partialSums[index];
  

  if(d_clust_recalcs[index]) return;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride >>=1) {
    if (index < stride) {

      shared_data[index].r += shared_data[index + stride].r;
      shared_data[index].g += shared_data[index + stride].g;
      shared_data[index].b += shared_data[index + stride].b;
    }
    __syncthreads();
  }

  if (index < k) {
    const int count = max(1, d_clust_sizes[index]);

    d_centroids[index].r = partialSums[index].r / count;
    d_centroids[index].g = partialSums[index].g / count;
    d_centroids[index].b = partialSums[index].b / count;
    partialSums[index] = {0, 0, 0};
    d_clust_sizes[index] = 0;
  }
}

__global__ void assignPoints(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int *assignments,
                             int *cluster_counts, int* recalc ,int k, int color_count) {

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
  
  if(assignments[idx]==closest_centroid){
    //recalc[closest_centroid]=1; 
  } else {
  assignments[idx] = closest_centroid;
   //recalc[closest_centroid]=0;
  }
  atomicAdd(&cluster_counts[closest_centroid], 1);
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
  int *d_clust_recalcs;
  cudaMalloc((void **)&d_colors, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_partialSums, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_random_points, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_assignments, totalPixels * sizeof(int));

  cudaMalloc((void **)&d_clusters, size * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_clust_recalcs, size * sizeof(int));
  cudaMalloc(&d_clust_sizes, size * sizeof(int));
  cudaMemcpy(d_colors, pixel_data, totalPixels * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  cudaMemset(d_assignments, -1, totalPixels * sizeof(int));
  cudaMemset(d_clust_recalcs, 0, size * sizeof(int));
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
  const int fine_shared_memory = THREADS_PER_BLOCK * sizeof(CUDA_COLOR_DATA);
  int x = 0;
  while (x != 5) {

    assignPoints<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_clusters, d_colors, d_assignments, d_clust_sizes, d_clust_recalcs ,size, totalPixels);

    sumClusters<<<blocksPerGrid, THREADS_PER_BLOCK, fine_shared_memory>>>(
        d_clusters, d_colors, d_assignments, d_clust_sizes, d_clust_recalcs,d_partialSums, size,
        totalPixels);

    recalcClusters<<<1, size *  blocksPerGrid, fine_shared_memory>>>(d_clusters, d_partialSums, d_clust_sizes, d_clust_recalcs, size);
    cudaMemset(d_clust_sizes, 0, size * sizeof(int));
    cudaMemset(d_partialSums, 0, size * sizeof(CUDA_COLOR_DATA));
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
  cudaFree(d_partialSums);

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
