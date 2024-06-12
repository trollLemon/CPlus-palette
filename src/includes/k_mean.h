/* *
 * k_mean.h
 *
 * Header for an implementation of the K_Means clustering algorithm
 * for color palette generation.
 *
 * This header includes definitions for funcitons to perform the algorithm,
 * along with wrappers and custom comparators to help with the execution
 *
 *
 *
 * */

#ifndef KMEAN
#define KMEAN
#include "color.h"
#include <vector>

  /* *
   * Returns the Euclidean Distance between two colors   
   * */
  double EuclideanDistance(Color *a, Color *b);


  /* *
   * Performs K means clustering for palette generation:
   * returns a vector of Color * for each color in the palette.
   * */
  std::vector<Color *> KMeans(std::vector<Color *> &colors, int k);

#endif
