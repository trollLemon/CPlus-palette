
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

#include "png++/png.hpp"
#include <string>

namespace pallet {

  png::image< png::index_pixel > getPixel (int width, int height);
  
  void makeColorPallet(std::string path, int size);

}

#endif
