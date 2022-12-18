

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
