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
