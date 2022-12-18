#include "color.h"

Color::Color(int r, int g, int b) : r{r}, g{g}, b{b} {}
std::string Color::asHex() {

  char hex[8];
    std::snprintf(hex, sizeof hex, "#%02x%02x%02x", r, g, b);

    std::string hexString;

    for (char i : hex) {
        hexString += i;
    }

    return hexString;

}

int Color::Red() {return r;}
int Color::Green() {return g;}
int Color::Blue() {return b;}

