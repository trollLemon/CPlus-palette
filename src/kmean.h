#ifndef KMEAN
#define KMEAN
#include <vector>
#include <array>
#include "dataTypes.h"

    /*
     * @Param p: reference to a point, q: reference to a point
     * @return: distance between two points
     *
     * Returns the distance (color distance) between two points
     */  
	double colorDistance(Point& p, Point& q);

    /* 
     * @param: clusters: reference to a vector of clusters, points: reference to a vector of points, k: number of clusters
     * @return: none
     * 
     * Creates clusters by randomly selecting k amoung of points from the points vector, and Creates clusters
     * with the selected points as centroids, and adds them to the clusters vector
     */
	void chooseCentroids(std::vector<Cluster>& clusters, std::vector<Point>& points, int k );

    /* 
     * @param: points: reference to a vector of points, clusters: reference to a vector of clusters
     * @return: none
     *
     * Compares the distance between each point and each centroid. Each point is added to the closest centroid.
     */
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters);

    /*   
     * @param: clusters: reference to a vector of clusters
     * @return: none
     *
     * Computes a new centroid for each cluster
     */
	void updateCentroids(std::vector<Cluster>& clusters);

    /*
     * @param: a: reference to an array of 3 ints, b: reference to an array of 3 ints
     * @return: bool
     *
     * returns true if the sum of the rgb values of each cluster *before* the alogrithm runs
     * is the same as the sum after the algorithm runs. This means the centroids have converged and we
     * can stop calculating new centroids
     */
	bool done(std::array<int,3>& a, std::array<int,3>& b);

    /*
     * @param: colorData: a reference to a vector of RGB values (stored in an array of 3 ints), size: size of color palette
     * @return: a vector of clusters
     *
     */
	std::vector<Cluster> generatePalette(std::vector<std::array<int,3>>& colorData, int size);

#endif
