/* *
 * Color.h
 *
 * Class to represent an RGB color.
 *
 * */

#ifndef COLOR
#define COLOR
#include <string>

class Color {

protected:
  // RGB Color Space
  unsigned char r;
  unsigned char g;
  unsigned char b;
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
  Color(unsigned char r, unsigned char g, unsigned char b);

  /* *
   * Returns the hex representation of this color
   *
   * */
  std::string asHex();



  /* *
   * Sets the RGB values for this color.
   * */
  void setRGB(unsigned char r, unsigned char g, unsigned char b);



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
 unsigned char Red() const;

  /* *
   * Returns the Green Value
   * */
   unsigned char Green() const;

  /* *
   * Returns the Blue value
   * */
  unsigned char Blue() const;
};

#endif
