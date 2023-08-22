#include "cuda_cluster.h"




__global__ void pointSummationKernel(ADV_Color** input,  double* output, int size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ float sharedMemory[];
	
	for (int i = 0; i< 3; /*we are concerned with 3 values, L, A, and B*/ ++i){	
		sharedMemory[threadIdx.x+i * blockDim.x] = 0.0;
	}



}


CudaCluster::CudaCluster(ADV_Color *a, int id): Cluster(a,id){};
void CudaCluster::calcNewCentroid(){

    std::vector<ADV_Color*> h_points = points;
    double* h_partialSum = new double[h_points.size() * 3];

    ADV_Color** d_points;
    cudaMalloc((void**)&d_points, h_points.size() * sizeof(ADV_Color*));

    cudaMemcpy(d_points, h_points.data(), h_points.size() * sizeof(ADV_Color*), cudaMemcpyHostToDevice);

    int blockSize = 256; // You can adjust this based on your GPU's capabilities
    int numBlocks = (h_points.size() + blockSize - 1) / blockSize;

    pointSummationKernel<<<numBlocks, blockSize>>>(d_points, h_partialSum, h_points.size());


}


