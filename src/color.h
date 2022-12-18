#ifndef COLOR
#define COLOR
#include <string>

class Color {

    friend class MedianCut;
    private:
        int r;
        int g;
        int b;

    public:
        Color(int r, int g, int b);
        std::string asHex();
        int Red();
        int Green();
        int Blue();
};



#endif
