#ifndef COLOR
#define COLOR
#include <cmath>
#include <string>

class Color {

private:
  // D65/2° standard illuminants for X Y and Z
  const double X_2 = 95.047;
  const double Y_2 = 100.000;
  const double Z_2 = 108.883;

  // RGB Color Space
  int r;
  int g;
  int b;

  // XYZ Color Space
  double x;
  double y;
  double z;
  // CIE Lab Color Space
  double L;
  double A;
  double B;

  // Color Space conversions
  void RGBtoXYZ();
  void XYZtoLAB();
  void LABtoXYZ();
  void XYZtoRGB();

  int clusterId; // clusterId
  int p;         // point id

public:
  bool operator<(const Color &ob) const;

  Color(int r, int g, int b, int p);
  void setLAB(double l, double a, double b);
  Color(double lum, double aVal, double bVal, int p);
  std::string asHex();
  int Red();
  int Green();
  int Blue();
  int getId();
  double Lum();
  double aVal();
  double bVal();
  void setClusterId(int i);
  int getClusterId();

  void testColor();

  // Convert RGB to LAB
  void RGBtoLAB();
  // converts LAB to RGB
  // Assumes that RGB was already converted to LAB
  void LABtoRGB();
};

#endif
