
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS
#define cimg_display 0
#include <string>

void makeColorPalette(std::string &path, int size, int genType);
/* *
 * Perform K_Mean clustering to generate a palette
 *
 * */
void K_Mean();

/* *
 * Performs median cut to generate a palette
 * */
void median_cut();

#endif
