#ifndef CUDA_CLUSTER
#define CUDA_CLUSTER
#include "cluster.h"


class CudaCluster: public Cluster {


public:
  CudaCluster(ADV_Color *a, int id);
void calcNewCentroid() override;

};
#endif

