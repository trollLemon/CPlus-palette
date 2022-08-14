
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

#include "png++/png.hpp"
#include <string>
#include <map>
#include <sstream>
namespace pallet {

  png::rgb_pixel  getPixel (int width, int height);
  
  void makeColorPallet(std::string path, int size);
  
  std::string  createHex(int r, int g, int b);

}

#endif
