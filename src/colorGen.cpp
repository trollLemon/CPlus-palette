#include "colorGen.h"
#include "CImg.h"
#include "color.h"
#include "median_cut.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
namespace palette {

using namespace cimg_library;



//check if user inputed palette size is a power of two
bool isPowerOfTwo(int x) {

    if(x == 1) return true;
    if(x==0) return false;
    return (x%2==0) && isPowerOfTwo(x/2);

}

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
            colors.push_back(new Color(r, g, b));
        }
    }
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

    MedianCut generator;
    std::vector<std::string> palette = generator.makePalette(colors, depth);


    for(int i = 0; i<tempSize; ++i){
        std::cout << palette[i] << std::endl;
    }

    for (auto i : colors) {
        delete i;
    }
}

} // namespace palette

