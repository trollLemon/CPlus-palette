/* *
 * ADV_Color.h
 *
 * Class to represent an advanced color for the K mean clustering algorithm.
 * This class will serve as the points for each cluster.
 *
 * The ADV_color inherits the basic color operations from the Color class, and
 * includes extended functionality:
 *      - LAB Color Space
 *      - XYZ Color Space
 *      - Conversion from RGB to LAB and vise versa
 *      - Cluster ID getting and setting
 *
 * The ADV_Color class has LAB color space variables so
 * differences in color can better match how the human eye
 * perceives color, view this wiki for more
 * information: https://en.wikipedia.org/wiki/CIELAB_color_space
 *
 * */

#ifndef ADVCOLOR
#define ADVCOLOR
#include "color.h"
#include <cmath>
#include <string>
class ADV_Color : public Color {

private:
  // D65/2° standard illuminants for X Y and Z
  const double X_2 = 95.047;
  const double Y_2 = 100.000;
  const double Z_2 = 108.883;

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

public:


  /* *
   * Constructor for an ADV_color with RGB values
   *
   * */
  ADV_Color(int r, int g, int b);

  /* *
   * Constructor for an ADV_color with LAB values
   *
   * */
  ADV_Color(double L, double A, double B);

#ifdef USE_CUDA
  ADV_Color& operator=(const ADV_Color& other);
#endif

  /* *
   * Sets the LAB values for this color.
   * The colors RGB values are updated to reflect the new
   * Color
   * */
  void setLAB(double l, double a, double b);


  /* *
   * Sets the RGB values for this color.
   * The colors LAB values are updated to reflect the new
   * Color
   * */
  void setRGB(int r, int g, int b);




  /* *
   * Returns the Luminance of the Color
   * */
  double Lum();

  /* *
   * Returns the Alpha value of the Color
   *
   * */
  double aVal();

  /* *
   * Returns the Beta Color of the Value
   * */
  double bVal();

  /* *
   * Converts the Current RGB values into LAB values.
   * Current LAB values will be overwritten
   * */
  void RGBtoLAB();

  /* *
   * Converts current LAB values to RGB values,
   * Current RGB values will be overwritten
   * */
  void LABtoRGB();

  /* *
   * Sets the cluster ID. This allows us to see which cluster the color is in.
   * If the cluster id is -1 then the color is assumed to not be in any cluster.
   *
   * */
  void setClusterId(int i);

  /* *
   * Returns the Id for the cluster the point is currently in
   *
   * */
  int getClusterId();
};

#endif
