
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

#include "CImg.h"
#include <string>

namespace pallet {

  
  void makeColorPallet(std::string path, int size);
  
  std::string  createHex(int r, int g, int b);

}

#endif
