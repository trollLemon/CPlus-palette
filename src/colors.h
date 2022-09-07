
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS
#define cimg_display 0
#include "CImg.h"
#include <string>

namespace palette {

  enum class paletteGenerationStatus 
  {
    success,
    imageLoadError,
    inputIsDirectory
  };

  paletteGenerationStatus  makeColorPalette(std::string& path, int size);
  
  std::string  createHex(int r, int g, int b);

}

#endif
