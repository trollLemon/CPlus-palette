
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

#include "png++/png.hpp"
#include <string>

namespace pallet {

  png::rgb_pixel  getPixel (int width, int height);
  
  void makeColorPallet(std::string path, int size);
  
  unsigned long createRGB(int r, int g, int b);

}

#endif
