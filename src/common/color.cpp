#include "color.h"



Color::Color() : r{0}, g{0}, b{0}, id{0} {}


Color::Color(unsigned char r, unsigned char g, unsigned char b) : r{r}, g{g}, b{b}, id(-1) {}

std::string Color::asHex() {
  char hex[9] = {'\0'};
  std::snprintf(hex, sizeof hex, "#%02x%02x%02x", r, g, b);

  std::string hexString(hex);

  return hexString;
}

/* *
 * Sets the RGB values for this color.
 * */
void Color::setRGB(unsigned char r, unsigned char g, unsigned char b) {
  this->r = r;
  this->g = g;
  this->b = b;
}

int Color::getClusterId() { return id; }
void Color::setClusterId(int i) { id = i; }
 unsigned char Color::Red() const { return r; }
 unsigned char Color::Green() const { return g; }
 unsigned char Color::Blue() const { return b; }
