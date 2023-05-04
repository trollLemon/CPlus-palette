/* *
 * Color.h
 *
 * Class to represent an RGB color.
 *
 * */

#ifndef COLOR
#define COLOR
#include <cmath>
#include <string>

class Color {

protected:
  // RGB Color Space
  int r;
  int g;
  int b;
  int id;
public:
  /* *
   * Custom comparator so we can compare points
   */
  bool operator<(const Color &ob) const;

  /* *
   * Constructor for a point class.
   * Requires an r , g and b value, along with a cluster id.
   *
   * */
  Color(int r, int g, int b);

  /* *
   * Returns the hex representation of this color
   *
   * */
  std::string asHex();



  /* *
   * Sets the RGB values for this color.
   * */
  void setRGB(double r, double g, double b);



  /* *
   * returns the points cluster id. 
   * We can see which cluster this point is in, 
   * so we can determine if a point moved or not
   *
   * */
int getClusterId();

/* *
 * sets the cluster id for this point.
 * if this point moves from one cluster to another, use this method
 * to reflect that
 *
 * */
void setClusterId(int i);


  /* *
   * Returns the R value
   * */
  int Red();

  /* *
   * Returns the Green Value
   * */
  int Green();

  /* *
   * Returns the Blue value
   * */
  int Blue();
};

#endif
