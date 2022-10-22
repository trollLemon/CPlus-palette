#include "CImg.h"
#include "dataTypes.h"
#include "kmean.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
namespace palette {

using namespace cimg_library;

// convert rgb values to hexidecimal
// gotten from lindevs: https://www.youtube.com/watch?v=TXMegco45q8
std::string createHex(int r, int g, int b) {

    char hex[8];
    std::snprintf(hex, sizeof hex, "#%02x%02x%02x", r, g, b);

    std::string hexString;

    for (char i : hex) {
        hexString += i;
    }

    return hexString;
}

/* This is what generates our color pallet.
 * We are going to go through every pixel and use a hashmap to keep
 * track of the frequency of the colors as we loop through each pixel.
 * Once we have a hashmap with colors and their frequencies, we will
 * sort it from most prominent color to least prominent, and then grab
 * the amount of colors dictated by the size variable.
 *

*/
void makeColorPalette(std::string &path, int size) {

    CImg<unsigned char> image(
        path.c_str()); // This is assigned if an image is loaded without errors,
                       // if not , then the program will exit and this wont be
                       // used

    int widthAndHeight{256};
    image.resize(widthAndHeight, widthAndHeight);
    int height{image.height()};
    int width{image.width()};

    // get the colors from each pixel

    std::vector<std::array<int, 3>> colors;

    for (int h{0}; h < height; ++h) {
        for (int w{0}; w < width; ++w) {
            std::array<int, 3> pixelColor = {0, 0, 0};
            pixelColor[0] = image(w, h, 0, 0);
            pixelColor[1] = image(w, h, 0, 1);
            pixelColor[2] = image(w, h, 0, 2);

            colors.push_back(pixelColor);
        }
    }

    std::vector<Cluster> palette{generatePalette(colors, size)};

    // sort palette based on the sum of the R G and B values
    // This is so we can sort the colors from darkest to brightest, since the
    // colors with low sums will be darker and vise versa
    std::sort(palette.begin(), palette.end(),
              [](Cluster &a, Cluster &b) { return a.getCentroid().sumRGB() > b.getCentroid().sumRGB(); });

    // reverse the order so its darkest colors to lightest
    std::reverse(palette.begin(), palette.end());

    for (Cluster& c : palette) {

        std::vector<int> rgb{c.getCentroid().getRGB()};

        std::cout << createHex(rgb[0], rgb[1], rgb[2]) << '\n';
    }
}

} // namespace palette

