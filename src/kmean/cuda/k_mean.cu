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
#include <iostream>
#include <random>
#include <unordered_set>
#include <atomic>

#define THREADS_PER_BLOCK 1024
#define max_diff 0.01
typedef adv_pixel CUDA_COLOR_DATA;


struct sum {

	double l;
	double a;
	double b;

};

__global__ void sumClusters(CUDA_COLOR_DATA *d_centroids,
                            CUDA_COLOR_DATA *d_colors, int *assignments,
                            int *d_clusterSizes, sum *partialSums,
                            int k, int color_count) {

  extern __shared__ sum shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (global_index >= color_count)
    return;

  int assn_cluster = assignments[global_index];

  double l, a, b;
  CUDA_COLOR_DATA color = d_colors[global_index]; 

  l = color.lab.a;
  a = color.lab.b;
  b = color.lab.c;

  sum blank = {0, 0, 0};
  sum col = {l,a,b};
  for (int cluster = 0; cluster < k; ++cluster) {

    shared_data[local_index] = (assn_cluster == cluster) ? col : blank;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (local_index < stride) {

        shared_data[local_index].l += shared_data[local_index + stride].l;
        shared_data[local_index].a += shared_data[local_index + stride].a;
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
                               sum *partialSums, int *d_clust_sizes,
                               int k, int* clusterConverged) {
  extern __shared__ sum shared_data[];

  const int index = threadIdx.x;
  shared_data[index] = partialSums[index];

  // if(d_clust_recalcs[index]) return;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride >>= 1) {
    if (index < stride) {

      shared_data[index].l += shared_data[index + stride].l;
      shared_data[index].a += shared_data[index + stride].a;
      shared_data[index].b += shared_data[index + stride].b;
    }
    __syncthreads();
  }

  if (index < k) {
    const int count = max(1, d_clust_sizes[index]);

    double prev_l = d_centroids[index].lab.a;
    double prev_a = d_centroids[index].lab.b;
    double prev_b = d_centroids[index].lab.c;

    d_centroids[index].lab.a = partialSums[index].l / count;
    d_centroids[index].lab.b = partialSums[index].a / count;
    d_centroids[index].lab.c = partialSums[index].b / count;
    

    double l = d_centroids[index].lab.a;
    double a = d_centroids[index].lab.b;
    double b = d_centroids[index].lab.c;

    double diff_l = fabs(l - prev_l);
    double diff_a = fabs(a - prev_a);
    double diff_b = fabs(b - prev_b);
    
    if (diff_l > max_diff && diff_a > max_diff && diff_b > max_diff)
	    atomicExch(&(*clusterConverged), 0); // the clusters have not converged yet

    partialSums[index] = {0, 0, 0};
    d_clust_sizes[index] = 0;
  }
  
}




__global__ void assignPoints(CUDA_COLOR_DATA *d_clusters,
                             CUDA_COLOR_DATA *d_colors, int *assignments,
                             int *cluster_counts, int k, int color_count) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ CUDA_COLOR_DATA s_clusters[];

  if (idx > color_count)
    return;

  int numThreadsPerBlock = blockDim.x;

  for (int i = threadIdx.x; i < k; i += numThreadsPerBlock) {
    s_clusters[i] = d_clusters[i];
  }

  __syncthreads();

  double min_dist = INFINITY;
  int closest_centroid = 0;

  CUDA_COLOR_DATA x = d_colors[idx];

  for (int i = 0; i < k; ++i) {

    CUDA_COLOR_DATA y = s_clusters[i];

    double dl = static_cast<double>(y.lab.a) - static_cast<double>(x.lab.a);
    double da = static_cast<double>(y.lab.b) - static_cast<double>(x.lab.b);
    double db = static_cast<double>(y.lab.c) - static_cast<double>(x.lab.c);
    

    double dist = dl * dl + da * da + db * db;
    dist = sqrt(dist);



    if (dist < min_dist) {
      min_dist = dist;
      closest_centroid = i;
    }
  }


  assignments[idx] = closest_centroid;
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

  

  sum *d_partialSums;

  int *d_assignments;
  int *d_random_points;
  int *d_clust_sizes;
  cudaMalloc((void **)&d_colors, totalPixels * sizeof(CUDA_COLOR_DATA));
  cudaMalloc((void **)&d_partialSums, totalPixels * sizeof(sum));
  cudaMalloc((void **)&d_random_points, totalPixels * sizeof(int));
  cudaMalloc((void **)&d_assignments, totalPixels * sizeof(int));

  cudaMalloc((void **)&d_clusters, size * sizeof(CUDA_COLOR_DATA));
  cudaMalloc(&d_clust_sizes, size * sizeof(int));
  cudaMemcpy(d_colors, pixel_data, totalPixels * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{0 , totalPixels};
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
  const int fine_shared_memory = THREADS_PER_BLOCK * sizeof(sum);
  int coarse_shared_memory =  size * blocksPerGrid * sizeof(sum);
  int sharedMemoryForClusters = size * sizeof(CUDA_COLOR_DATA);
  int h_clusterConverged = 0;  
  int initGpuValue = 1;
  int* clusterConverged;
  cudaMalloc((void**)&clusterConverged, sizeof(int));
cudaMemcpy(clusterConverged, &initGpuValue, sizeof(int), cudaMemcpyHostToDevice);


  while (!h_clusterConverged) {
   
    h_clusterConverged = 0;
  cudaMemcpy(clusterConverged, &clusterConverged, sizeof(int), cudaMemcpyHostToDevice);

    assignPoints<<<blocksPerGrid, THREADS_PER_BLOCK, sharedMemoryForClusters>>>(d_clusters, d_colors, d_assignments, d_clust_sizes, size, totalPixels);

    sumClusters<<<blocksPerGrid, THREADS_PER_BLOCK, fine_shared_memory>>>(d_clusters, d_colors, d_assignments, d_clust_sizes, d_partialSums, size,totalPixels);

    recalcClusters<<<1, blocksPerGrid, coarse_shared_memory>>>( d_clusters, d_partialSums, d_clust_sizes, size, clusterConverged);

    cudaDeviceSynchronize();
    cudaMemset(d_clust_sizes, 0, size * sizeof(int));
    cudaMemset(d_partialSums, 0, size * sizeof(CUDA_COLOR_DATA));
    cudaMemcpy(&h_clusterConverged, clusterConverged, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << h_clusterConverged << std::endl;
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
  cudaFree(clusterConverged);
  ADV_Color color_helper(0, 0, 0);
  std::vector<std::string> palette;
  for (int i = 0; i < size; ++i) {

    CUDA_COLOR_DATA col = h_colors[i];
    
    non_rgb_colorspace lab = col.lab;
    non_rgb_colorspace xyz = lab_to_xyz(lab.a,lab.b,lab.c);


    pixel color = xyz_to_rgb(xyz.a,xyz.b,xyz.c);

    color_helper.setRGB(color.r, color.g, color.b);

    palette.push_back(color_helper.asHex());
  }

  free(h_colors);

  return palette;
}
