#ifndef THE_DATA_TYPES
#define THE_DATA_TYPES
#include <vector>



struct Point {
	int r;
	int g;
	int b;

	int id;
		Point(int red, int green, int blue);
		Point();
		int sumRGB();

	};
	
	
	struct Cluster {
  
  Point centroid;
	std::vector<Point> data;
  int id;
	
	Cluster(Point p, int id);

	void addPoint(Point p);
	void setCentroid(Point p);
	std::vector<Point>& getData();
	void resetCentroid();

	};
	

#endif
