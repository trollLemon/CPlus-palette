
#include "adv_color.h"
#include <cmath>
#include <iostream>
ADV_Color::ADV_Color(int r, int g, int b) : Color(r, g, b) {

  // convert to XYZ and LAB on color creation
  RGBtoLAB();
}

ADV_Color::ADV_Color(double lum, double aVal, double bVal) :  Color(0, 0, 0), L{lum}, A{aVal}, B{bVal} {

  // convert to XYZ and RGB on color creation
  LABtoRGB();
}

// https://www.easyrgb.com/en/math.php
void ADV_Color::RGBtoXYZ() {

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
void ADV_Color::XYZtoLAB() {

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
void ADV_Color::LABtoXYZ() {

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
void ADV_Color::XYZtoRGB() {

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
void ADV_Color::RGBtoLAB() {

  RGBtoXYZ();
  XYZtoLAB();
}
// converts LAB to RGB
// Assumes that RGB was already converted to LAB
void ADV_Color::LABtoRGB() {

  LABtoXYZ();
  XYZtoRGB();
}

void ADV_Color::setLAB(double l, double a, double b) {
  this->L = l;
  this->A = a;
  this->B = b;
  LABtoRGB();
}

void ADV_Color::setRGB(int r, int g, int b) {
  this->r = r;
  this->g = g;
  this->b = b;
  RGBtoLAB();
}

double ADV_Color::Lum() { return L; }
double ADV_Color::aVal() { return A; }
double ADV_Color::bVal() { return B; }

/* *
 * Sets the cluster ID. This allows us to see which cluster the color is in.
 * If the cluster id is -1 then the color is assumed to not be in any cluster.
 *
 * */
void ADV_Color::setClusterId(int i) { clusterId = i; }

/* *
 * Returns the Id for the cluster the point is currently in
 *
 * */
int ADV_Color::getClusterId() { return clusterId; }
