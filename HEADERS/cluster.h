/*  Cluster.h
 *  Header file for a Cluster Class.
 *  A Cluster contains a list of points, one of them being the centroid.
 *  The centroid is the average value of all the other points within the
 * cluster. A cluster is also assigned a unique ID.
 *
 *  This implementation provides client code the ability to add points to a
 * cluster, get the clusters id, get the hex value of the centroids color, and
 * recalculate a new centroid.
 *
 * */

#ifndef CLUSTER
#define CLUSTER
#include "adv_color.h"
#include <vector>

class Cluster {

private:
  ADV_Color *centroid;
  std::vector<ADV_Color *> points;
  int id;

public:
  /* *
   * Returns the clusters centroid
   * If no centroid was recalculated by other code prior to calling
   * this function, the initial centroid is returned
   * */
  ADV_Color *getCentroid() const;

  /* *
   * Constructor for a Cluster class.
   * The constructor requires a color pointer to be
   * the initial centroid, and an Id so we can identify
   * the cluster later.
   *
   * */
  Cluster(ADV_Color *a, int id);

  /* *
   * Default Destructor
   *
   * Destructor only frees memory used by the Cluster class, the clusters
   * points are deallocated
   * in another section of the program
   *
   * */
  ~Cluster();

  /* *
   * Calculated a new centroid from the Clusters points.
   * Assumes other points have been added to the cluster, otherwise nothing will
   * happen.
   *
   * This method "replaces" the old centroid by overwriting the old centroids
   * LAB values with updated values.
   *
   * The Clusters points are cleared once the new centroid is calculated.
   * */
  void calcNewCentroid();

  /* *
   * Returns the clusters Id
   *
   * */
  int getId();

  /* *
   * Returns the hex representation for the centroids RGB values.
   * */
  std::string asHex();

  /* *
   * Adds a point to the Cluster
   * Assumes the point hasn't been added to the cluster already in the
   * current iteration of K mean clustering
   * */
  void addPoint(ADV_Color *points);
};

#endif
