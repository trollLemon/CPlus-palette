
#ifndef PALETTEGEN
#define PALETTEGEN

#include <vector>
#include <array>
#include <tuple>
namespace kmean_cluster {
static double maxDiff = 10.0;
 
	struct Point {

		unsigned char r;
		unsigned char g;
		unsigned char b;

	};

	struct Cluster {

		std::vector<Point> points;
		Point centroid;
	};

	
	bool sortBySecond(std::tuple<Cluster, double> a, std::tuple<Cluster, double> b);

	std::vector<Point> makePointsFromImageData (std::vector<std::array<unsigned char, 3>> colorData);
	
	Point makePoint (std::array<unsigned char, 3 > rgb);

	Cluster makeCluster (Point randomPoint);
	
	std::vector<Point> makePalette (std::vector<std::array<unsigned char ,3>> colorData, int size);
	
	double distance(Point p, Point q);
	
	Point calculateCentriod(Cluster cluster);

	std::array<double,3> centriodDifference(Cluster a, Cluster b);

}

#endif
