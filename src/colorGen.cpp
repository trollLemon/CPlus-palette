#include "CImg.h"
#include <algorithm>
#include "color.h"
#include "colorGen.h"
#include "median_cut.h"
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

    int widthAndHeight{500};
    image.resize(widthAndHeight, widthAndHeight);
    int height{image.height()};
    int width{image.width()};

    // get the colors from each pixel

    std::vector<Color *> colors;

    for (int h{0}; h < height; ++h) {
        for (int w{0}; w < width; ++w) {
            int r = image(w, h, 0, 0);
            int g = image(w, h, 0, 1);
            int b = image(w, h, 0, 2);
            colors.push_back(new Color(r,g,b));
        }
    }
    

    int depth = log2(static_cast<double>(size));

    MedianCut generator;
    std::vector<std::string> palette = generator.makePalette(colors, depth);
}

} // namespace palette


