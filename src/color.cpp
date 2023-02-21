#include "color.h"
#include <cmath>
#include <iostream>
Color::Color(int r, int g, int b, int p) : r{r}, g{g}, b{b}, p{p} {

  // convert to XYZ and LAB on color creation
  RGBtoLAB();
}
std::string Color::asHex() {

  char hex[8];
  std::snprintf(hex, sizeof hex, "#%02x%02x%02x", r, g, b);

  std::string hexString;

  for (char i : hex) {
    hexString += i;
  }

  return hexString;
}

// https://www.easyrgb.com/en/math.php
void Color::RGBtoXYZ() {

  double ratioR = (double)r / (255.0);
  double ratioG = (double)g / (255.0);
  double ratioB = (double)b / (255.0);

  if (ratioR > 0.04045) {
    ratioR = std::pow((ratioR + 0.055) / 1.055, 2.4);
  } else {
    ratioR /= 12.92;
  }

  if (ratioG > 0.04045) {
    ratioG = std::pow((ratioG + 0.055) / 1.055, 2.4);
  } else {
    ratioG /= 12.92;
  }

  if (ratioB > 0.04045) {
    ratioB = std::pow((ratioB + 0.055) / 1.055, 2.4);
  } else {
    ratioB /= 12.92;
  }

  ratioR *= 100.0;
  ratioG *= 100.0;
  ratioB *= 100.0;

  x = ratioR * 0.4124 + ratioG * 0.3576 + ratioB * 0.1805;
  y = ratioR * 0.2126 + ratioG * 0.7152 + ratioB * 0.0722;
  z = ratioR * 0.0193 + ratioG * 0.1192 + ratioB * 0.9505;
}
void Color::XYZtoLAB() {

  double ratioX = x / (X_2);
  double ratioY = y / (Y_2);
  double ratioZ = z / (Z_2);

  if (ratioX > 0.008856) {
    ratioX = std::pow(ratioX, (1.0 / 3.0));
  } else {
    ratioX = (7.787 * ratioX) + (16.0 / 116.0);
  }

if (ratioY > 0.008856) {
    ratioY = std::pow(ratioY, (1.0 / 3.0));
  } else {
    ratioY = (7.787 * ratioY) + (16.0 / 116.0);
  }

if (ratioZ > 0.008856) {
    ratioZ = std::pow(ratioZ, (1.0 / 3.0));
  } else {
    ratioZ = (7.787 * ratioZ) + (16.0 / 116.0);
  }



L = (116.0 * ratioY) - 16.0;
A = 500.0 * (ratioX-ratioY);
B = 200.0 * (ratioY-ratioZ);

}
void Color::LABtoXYZ() {}
void Color::XYZtoRGB() {}

// Convert RGB to LAB
void Color::RGBtoLAB() {

  RGBtoXYZ();
  XYZtoLAB();
}
// converts LAB to RGB
// Assumes that RGB was already converted to LAB
void Color::LABtoRGB() {}
void Color::testColor() {

  std::cout << "Color:" << p << std::endl;

  std::cout << "RGB: " << r << " " << g << " " << b << std::endl;
  std::cout << "LAB: " << L << " " << A << " " << B << std::endl;
}
int Color::getId() { return p; }
int Color::getClusterId() { return ClusterId; }
void Color::setClusterId(int i) { ClusterId = i; }
int Color::Red() { return r; }
int Color::Green() { return g; }
int Color::Blue() { return b; }
double Color::Lum() { return L; }
double Color::aVal() { return A; }
double Color::bVal() { return B; }
