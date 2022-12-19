
#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS
#define cimg_display 0
#include <string>

namespace palette {
    /* 
     * @Param: string path:Image path, int size: size of color palette
     * @return: none
     *
     * Makes a color palette
     */
    void makeColorPalette(std::string &path, int size);
}

#endif
