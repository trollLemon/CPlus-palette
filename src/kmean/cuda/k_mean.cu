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
#include "util.h"

typedef pixel CUDA_COLOR_DATA;

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

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx>color_count) return;

	double min_dist = INFINITY;
	int closest_centroid = 0;


	for(int i = 0; i<k; ++i){
	 double dist = euclideanDistance(d_clusters[i],d_colors[idx]);

	 if (dist<min_dist){
		min_dist =dist;
		closest_centroid =i;
	 }

	}

	assignments[idx]=closest_centroid;
   atomicAdd(&cluster_stats[closest_centroid], 1);
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
CudaKmeanWrapper(CUDA_COLOR_DATA* pixel_data, int size, int totalPixels) {

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
  cudaMemcpy(d_colors, pixel_data, totalPixels * sizeof(CUDA_COLOR_DATA),
             cudaMemcpyHostToDevice);

  bool loop = true;
  int threadsPerBlock = 32;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  std::random_device rd;
  int seed = rd();
  initClusters<<<blocksPerGrid, threadsPerBlock>>>(d_clusters, d_colors, size,
                                                   seed);
  cudaDeviceSynchronize();
 int x =0;
  while(x++!=5){

  blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
  assignPoints<<<blocksPerGrid, threadsPerBlock>>>(
      d_clusters, d_colors, d_assignments, size, totalPixels, d_cluster_stats);
  recalcClusters<<<blocksPerGrid, threadsPerBlock>>>(
    d_clusters, d_colors, d_assignments, totalPixels, d_cluster_stats);

  cudaDeviceSynchronize();

  cudaMemset(d_cluster_stats, 1, size * sizeof(int));

  //}
  }

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
