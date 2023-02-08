#ifndef COLOR
#define COLOR
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
  int x;
  int y;
  int z;
  // CIE Lab Color Space
  int L;
  int A;
  int B;

  void RGBtoXYZ();
  void XYZtoLAB();

  int ClusterId;
  int p;
public:
  bool operator<(const Color &ob) const;

  Color(int r, int g, int b, int p);
  std::string asHex();
  int Red();
  int Green();
  int Blue();
  void setClusterId();
  void getClusterId();
  void RGBtoLAB();
  void LABtoRGB();
};

#endif
