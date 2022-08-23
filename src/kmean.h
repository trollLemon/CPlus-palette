
#ifndef PALETTEGEN
#define PALETTEGEN

#include <vector>
#include <array>
namespace kmean_cluster {
static double maxDiff = 20.0;
 
	struct Point {

		unsigned char r;
		unsigned char g;
		unsigned char b;

	};

	struct Cluster {

		std::vector<Point> points;
		Point centriod;
	};


	std::vector<Point> makePointsFromImageData (std::vector<std::array<unsigned char, 3>> colorData);
	
	Point makePoint (std::array<unsigned char, 3 > rgb);

	Cluster makeCluster (Point randomPoint);
	
	std::vector<Point> makePalette (std::vector<std::array<unsigned char ,3>> colorData);
	
	double distance(Point p, Point q);
	
	Point calculateCentriod(Cluster cluster);

}

#endif
