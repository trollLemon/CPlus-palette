#ifndef THE_DATA_TYPES
#define THE_DATA_TYPES
#include <vector>

// Class for a point object, which will be the color data for each pixel in the
// image
class Point {
  private:
    int r;  // red value
    int g;  // green value
    int b;  // blue value
    int id; // used to show which cluster the point is in

  public:
    /*
     * @param red: red value, green: green value, blue: blue value
     * @returns: nothing
     *
     * Created a point object with the RGB data. Id is set to -1 if the point
     * isnt in a cluster.
     */
    Point(int red, int green, int blue);

    Point(); // default constructor

    /*
     * @param: none
     * @return: id of Point
     *
     * Returns the id of the Point
     */
    int getId() const;

    /*
     *@param: id: int denoting which cluster the point is in
     *@return: nothing
     */
    void setId(int id);

    /*
     * @param: none
     * @returns: a vector of the RGB values (ints) in the Point
     */
    std::vector<int> getRGB();

    /*
     * @param: none
     * @return: The sum of the R G and B values in the point
     */
    int sumRGB();
};

/* Class for a data cluster.
 * The K mean clustering algorithm uses clusters, which include:
 *
 *            - A centroid (center or average value)
 *            - A List of Points including the centroid
 *
 * A cluster will recompute a new centriod for each iteration of the alogrithm.
 */
class Cluster {

  private:
    Point centroid;          // average value for data
    std::vector<Point> data; // list of Points
    int id;                  // used to identify the cluster

  public:
    /* Assuming data was loaded from the image and put into Point objects:
     *
     * @param: p: a Point object, id: identification for the cluster
     * @return: nothing
     */
    Cluster(Point &p, int id);

    /* Assuming points were loaded from the image and put into Point objects
     *
     * @param: p: a reference to a point object
     * @return: nothing
     */
    void addPoint(Point &p);

    /*
     * @param: none
     * @return: the id of the cluster
     */
    int getId() const;

    /*
     * Assuming the cluster has data in it:
     *
     * @param: none
     * @return: nothing
     *
     * Averages all points within the cluster, and sets the clusters centroid
     * to the new calulated centroid.
     */
    void calculateNewCentroid();

    /* Assuming the Cluster has data in it:
     *
     * @param: none
     * @return: a reference to a vector of Point objects
     *
     */
    std::vector<Point>& getData();

    /*
     * @param:none
     * @return: nothing
     *
     * Clears the data in the cluster (centroid stays however)
     */
    void reset();
    
    /*  
     * @param: none
     * @return reference to the centroid
     *
     */
    Point &getCentroid();
};

#endif
