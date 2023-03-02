#include "colorGen.h"
#include "CImg.h"
#include "color.h"
#include "quantizer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
namespace palette {

using namespace cimg_library;

void makeColorPalette(std::string &path, int size) {

  CImg<unsigned char> image(
      path.c_str()); // This is assigned if an image is loaded without errors,
                     // if not , then the program will exit and this wont be
                     // used

  int widthAndHeight{200};
  image.resize(widthAndHeight, widthAndHeight);
  int height{image.height()};
  int width{image.width()};

  // get the colors from each pixel
  int count = 0;
  std::vector<Color *> colors;

  for (int h{0}; h < height; ++h) {
    for (int w{0}; w < width; ++w) {
      int r = image(w, h, 0, 0);
      int g = image(w, h, 0, 1);
      int b = image(w, h, 0, 2);
      Color *newColor = new Color(r, g, b, count++);
      colors.push_back(newColor);
    }
  }

  Quantizer q;

  std::vector<std::string> palette = q.makePalette(colors, size);

  for (std::string color : palette) {
    std::cout << color << std::endl;
  }

  for (Color *color : colors) {
    delete color;
  }
}

} // namespace palette
