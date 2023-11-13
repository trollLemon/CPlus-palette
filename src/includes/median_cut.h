/* *
 * median_cut.h
 *
 * Header for an implementation of median cut for color palette generation.
 *
 * Unlike the K_Mean implementation, we only need basic RGB colors.
 *
 * */

#ifndef MEDIANCUT
#define MEDIANCUT
#include "color.h"
#include <string>
#include <vector>

  /* *
   * recursively sorts colors based on the range of 
   * RGB values, then splits them into two buckets.
   *
   * The recursion stops once we reach a certain depth
   * in the process, which results in the correct amount of buckets
   * for the color palette
   *
   * */
  void mc(std::vector<Color *> &colors, std::vector<std::string> &palette, int k);
  /* *
   * Returns the average color of a bucket containing points
   * assumes the bucket has more than 0 points in it
   * */
  void getAverageColor(std::vector<Color *> &colors, std::vector<std::string> &palette);

  

  /* *
   * Returns a vector containing the hex representations of each color in 
   * the color palette. 
   *
   * Requires a vector of color pointers, and a depth k to limit the recursion to the 
   * required number of colors
   *
   * */
  std::vector<std::string> median_cut(std::vector<Color *> &colors, int k);

#endif
