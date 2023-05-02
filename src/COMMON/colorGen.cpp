#include "colorGen.h"
#include "CImg.h"
#include "color.h"
#include "median_cut.h"
#include "quantizer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <string>
#include <unordered_set>
#include <vector>
namespace palette {

using namespace cimg_library;


// check if user inputed palette size is a power of two
bool isPowerOfTwo(int x) {

  if (x == 1)
    return true;
  if (x == 0)
    return false;
  return (x % 2 == 0) && isPowerOfTwo(x / 2);
}

void makeColorPalette(std::string &path, int size, int genType) {

  CImg<unsigned char> image(
      path.c_str()); // This is assigned if an image is loaded without errors,
                     // if not , then the program will exit and this wont be
                     // used

  int widthAndHeight{500};
  image.resize(widthAndHeight, widthAndHeight);
  int height{image.height()};
  int width{image.width()};

  // get the colors from each pixel
  int count = 0;
  
  // The colors from the image
  std::vector<Color *> colors;
  

  // we will use a map to filter out duplicate colors
  // for example, an image that is 90% dark blue will likely pollute 
  // the other colors during the clustering, so we only include the color once
  // this improves color palette generation by a lot 
  std::unordered_set<std::string> unique;
  
  for (int h{0}; h < height; ++h) {
    for (int w{0}; w < width; ++w) {
      int r = image(w, h, 0, 0);
      int g = image(w, h, 0, 1);
      int b = image(w, h, 0, 2);
      Color *newColor = new Color(r, g, b, count++);

      if (unique.count(newColor->asHex()) != 0) {
        delete newColor;
      } else {
        unique.insert(newColor->asHex());
        colors.push_back(newColor);
      }
    }
  }

  std::vector<std::string> palette;

  switch (genType) {
  case 1: {
    std::cout << "Using K Mean Clustering:::" << std::endl;
    Quantizer q;
    palette = q.makePalette(colors, size);
    for (std::string color : palette) {
      std::cout << color << std::endl;
    }

    break;
  }
  case 2: {
    std::cout << "Using MedianCut:::" << std::endl;
    MedianCut generator;
    int tempSize = size;
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
    palette = generator.makePalette(colors, depth);
    for (int i = 0; i < tempSize; ++i) {
      std::cout << palette[i] << std::endl;
    }
    break;
  }
  }

  for (Color *color : colors) {
    delete color;
  }
}

} // namespace palette
