#include "median_cut_helpers.h"


bool isPowerOfTwo(int size){
return false;

}

int powerOfTwoSize(int size){
if (!isPowerOfTwo(size)) {
      size--;
      size |= size >> 1;
      size |= size >> 2;
      size |= size >> 4;
      size |= size >> 8;
      size |= size >> 16;
      size++;
    }
    int depth = log2(static_cast<double>(size));
}
