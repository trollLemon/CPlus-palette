#ifndef KMEAN
#define KMEAN
#include <vector>
#include <array>
#include "dataTypes.h"

	double colorDistance(Point& p, Point& q);
	void chooseCentroids(std::vector<Cluster>& clusters, std::vector<Point>& points, int k );
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters);
	void updateCentroids(std::vector<Cluster>& clusters);

	bool done(std::array<int,3>& a, std::array<int,3>& b);

	std::vector<Point> generatePalette(std::vector<std::array<int,3>>& colorData, int size);

#endif
