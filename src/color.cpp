#include "color.h"
#include <cmath>
#include <iostream>
Color::Color(int r, int g, int b, int p)
    : r{r}, g{g}, b{b}, p{p}, clusterId(-1) {

  // convert to XYZ and LAB on color creation
  RGBtoLAB();
}

Color::Color(double lum, double aVal, double bVal, int p)
    : L{lum}, A{aVal}, B{bVal}, p{p}, clusterId(-1) {

  // convert to XYZ and RGB on color creation
  LABtoRGB();
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
  A = 500.0 * (ratioX - ratioY);
  B = 200.0 * (ratioY - ratioZ);
}
void Color::LABtoXYZ() {

  double ratioY = (L + 16.0) / 116.0;
  double ratioX = A / 500.0 + ratioY;
  double ratioZ = ratioY - B / 200.0;

  if (std::pow(ratioY, 3.0) > 0.008856) {
    ratioY = std::pow(ratioY, 3.0);
  } else {
    ratioY = (ratioY - 16.0 / 116.0) / 7.787;
  }

  if (std::pow(ratioX, 3.0) > 0.008856) {
    ratioX = std::pow(ratioX, 3.0);
  } else {
    ratioX = (ratioX - 16.0 / 116.0) / 7.787;
  }

  if (std::pow(ratioZ, 3.0) > 0.008856) {
    ratioZ = std::pow(ratioZ, 3.0);
  } else {
    ratioZ = (ratioZ - 16.0 / 116.0) / 7.787;
  }

  x = ratioX * X_2;
  y = ratioY * Y_2;
  z = ratioZ * Z_2;
}
void Color::XYZtoRGB() {

  double ratioX = x / 100.0;
  double ratioY = y / 100.0;
  double ratioZ = z / 100.0;

  double ratioR = ratioX * 3.2406 + ratioY * -1.5372 + ratioZ * -0.4986;
  double ratioG = ratioX * -0.9689 + ratioY * 1.8758 + ratioZ * 0.0415;
  double ratioB = ratioX * 0.0557 + ratioY * -0.2040 + ratioZ * 1.0570;

  if (ratioR > 0.0031308) {
    ratioR = 1.055 * std::pow(ratioR, (1.0 / 2.4)) - 0.055;
  } else {
    ratioR = 12.92 * ratioR;
  }

  if (ratioG > 0.0031308) {
    ratioG = 1.055 * std::pow(ratioG, (1.0 / 2.4)) - 0.055;
  } else {
    ratioG = 12.92 * ratioG;
  }

  if (ratioB > 0.0031308) {
    ratioB = 1.055 * std::pow(ratioB, (1.0 / 2.4)) - 0.055;
  } else {
    ratioB = 12.92 * ratioB;
  }

  r = ratioR * 255;
  g = ratioG * 255;
  b = ratioB * 255;
}

// Convert RGB to LAB
void Color::RGBtoLAB() {

  RGBtoXYZ();
  XYZtoLAB();
}
// converts LAB to RGB
// Assumes that RGB was already converted to LAB
void Color::LABtoRGB() {

  LABtoXYZ();
  XYZtoRGB();
}
void Color::testColor() {

  std::cout << "Color:" << p << std::endl;

  std::cout << "RGB: " << r << " " << g << " " << b << std::endl;
  std::cout << "XYZ: " << x << " " << y << " " << z << std::endl;
  std::cout << "LAB: " << L << " " << A << " " << B << std::endl;
}

void Color::setLAB(double l, double a, double b) {
  this->L = l;
  this->A = a;
  this->B = b;
  LABtoRGB();
}

int Color::getId() { return p; }
int Color::getClusterId() { return clusterId; }
void Color::setClusterId(int i) { clusterId = i; }
int Color::Red() { return r; }
int Color::Green() { return g; }
int Color::Blue() { return b; }
double Color::Lum() { return L; }
double Color::aVal() { return A; }
double Color::bVal() { return B; }
