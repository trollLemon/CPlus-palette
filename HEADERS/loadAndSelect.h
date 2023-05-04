
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS
#define cimg_display 0
#include <string>
#include "CImg.h"


using namespace cimg_library;
/* *
 * Creates a color palette from the given path to an image.
 * Requires an image path, a palette size, and a generation type.
 * 1 = KMeans 
 * 2 = Median Cut
 * */

void makeColorPalette(std::string &path, int size, int genType);

#endif
