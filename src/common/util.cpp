#include "util.h"
#include "cluster.h"
#include <cmath>
cluster_distance::~cluster_distance() {}
void minHeap::push(cluster_distance *pair) {

  pair->distance *= -1;

  distances.push(pair);
}

bool Comp::operator()(const cluster_distance *a, const cluster_distance *b) {
  return a->distance < b->distance;
}

minHeap::~minHeap() { clear(); }

int minHeap::pop() {

  if (distances.top() == nullptr) {
    return -1;
  }

  int ClusterId = distances.top()->cluster;
  delete distances.top();
  distances.pop();

  return ClusterId;
}
void minHeap::clear() {

  while (!distances.empty()) {
    delete distances.top();
    distances.pop();
  }
}

non_rgb_colorspace rgb_to_xyz(int r, int g, int b) {
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

  double x = ratioR * 0.4124 + ratioG * 0.3576 + ratioB * 0.1805;
  double y = ratioR * 0.2126 + ratioG * 0.7152 + ratioB * 0.0722;
  double z = ratioR * 0.0193 + ratioG * 0.1192 + ratioB * 0.9505;
  return {x,y,z};
}
non_rgb_colorspace xyz_to_lab(double x, double y, double z) {
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

  double l = (116.0 * ratioY) - 16.0;
  double a = 500.0 * (ratioX - ratioY);
  double b = 200.0 * (ratioY - ratioZ);
  return {l,a,b};
}
non_rgb_colorspace lab_to_xyz(double l, double a, double b) {


  double ratioY = (l + 16.0) / 116.0;
  double ratioX = a / 500.0 + ratioY;
  double ratioZ = ratioY - b / 200.0;

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

  double x = ratioX * X_2;
  double y = ratioY * Y_2;
  double z = ratioZ * Z_2;

  return {x,y,z};
}
pixel xyz_to_rgb(double x, double y, double z) {
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

  int r = ratioR * 255;
  int g = ratioG * 255;
  int b = ratioB * 255;
  return {r,g,b};
}
