#include <iostream>
#include "k_mean_cuda.h"
#include <cuda_runtime.h>

#include "cuda_cluster.h"
#include "color.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>


double CUDAKMean::EuclidianDistance(ADV_Color *a, ADV_Color *b) {

  double deltaL = a->Lum() - b->Lum();
  double deltaA = a->aVal() - b->aVal();
  double deltaB = a->bVal() - b->bVal();

  return std::sqrt((deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB));
}



__global__ void initPoints(ADV_Color** d_points){
int tid = threadIdx.x + blockIdx.x * blockDim.x;

}


__global__ void recalcDistance(ADV_Color** d_colors,Cluster** d_clusters){
int tid = threadIdx.x + blockIdx.x * blockDim.x;


}


void CUDAKMean::K_MEAN_START() {

  // recalculate distances for all points
  // if any points move, we will put the effected clusters in a set,

  std::set<Cluster *> toRecalculate;

  bool loop = true;


  while (loop || toRecalculate.size() != 0) {

    toRecalculate.clear();
    for (ADV_Color *point : colors) {

      for (auto cluster : clusters) {

        double distance =
            EuclidianDistance(cluster.second->getCentroid(), point);
        int id = cluster.second->getId();
      }

    }
    if (toRecalculate.size() == 0) {
      return;
    }
    for (Cluster *cluster : toRecalculate) {
      cluster->calcNewCentroid();
    }

    loop = !loop;
  }
}


void CUDAKMean::K_MEAN_INIT(int k) {

	
  // randomly get k points
  int size = colors.size();

  // better seeding from
  // https://www.learncpp.com/cpp-tutorial/generating-random-numbers-using-mersenne-twister/
  std::random_device rd;
  std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};

  std::mt19937 mt{ss};
  std::uniform_int_distribution<> kPoints{size / 2, size};
  std::set<int> seen; // make sure we have unique numbers
  std::vector<int> colorIndecies;
  while (seen.size() != static_cast<long unsigned int>(k)) {
    int num = kPoints(mt);
    if (seen.count(num) == 0) {
      seen.insert(num);
      colorIndecies.push_back(num);
    }
  }

  for (int i = 0; i < k; ++i) {
    Cluster *cluster = new Cluster(colors[colorIndecies[i]], i);
    clusters[i] = cluster;
  }

    ADV_Color** d_points;
    cudaMalloc((void**)&d_points, colors.size() * sizeof(ADV_Color*));
    cudaMemcpy(d_points, colors.data(), colors.size() * sizeof(ADV_Color*), cudaMemcpyHostToDevice);
    int blockSize = 256; 
    int numBlocks = (colors.size() + blockSize - 1) / blockSize;
    initPoints<<<numBlocks,blockSize>>>(d_points);
    cudaMemcpy(d_points, colors.data(), colors.size() * sizeof(ADV_Color*), cudaMemcpyDeviceToHost);
    cudaFree(d_points);


    // first phase done


}

std::vector<std::string> CUDAKMean::makePalette(std::vector<ADV_Color *> &colors,
                                            int k) {


	this->colors = colors;
	K_MEAN_INIT(k);






  return {};
}










std::vector<std::string> CudaKmeanWrapper(cimg_library::CImg<unsigned char> *image, int size){
  std::vector<ADV_Color *> colors;
  ADV_Color *base = new ADV_Color(0, 0, 0);
int height{image->height()};
  int width{image->width()};
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);
      colors.push_back(new ADV_Color(r, g, b));
  }
  }

  	CUDAKMean *kmean = new CUDAKMean();
	std::cout <<"Using GPU for Palette Generation...." << std::endl;
	std::vector<std::string> palette = kmean->makePalette(colors, size);

	return palette;
    
}


