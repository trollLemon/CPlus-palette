

//    Move all pixels into a single large bucket.
//    Find the color channel (red, green, or blue) in the image with the greatest range.
//    Sort the pixels by that channel values.
//    Find the median and cut the region by that pixel.
//    Repeat the process for both buckets until you have the desired number of colors.


#ifndef MEDIANCUT
#define MEDIANCUT
#include <vector>
#include <string>
#include "color.h"
class MedianCut {
    
    
    private:
    void median_cut(std::vector<Color *> colors, int k);
    std::vector<std::string> colors;
    std::string getAverageColor(std::vector<Color *> colors);
    public:
    std::vector<std::string> makePalette(std::vector<Color *> colors, int k);





};

#endif
