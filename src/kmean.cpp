#include <vector>
#include <array>
#include "kmean.h"
namespace kmean_cluster {


	Point makePoint (std::array<unsigned char, 3 > rgb)
	{
		Point currPoint;

		currPoint.r = rgb[0];
		currPoint.g =  rgb[1];
		currPoint.b = rgb[2];

		return currPoint;
	}

	Cluster makeCluster (Point randomPoint)
	{
		std::vector<Point> points;

		points.push_back(randomPoint);//add the centroid to the list of points so every cluster has at least 1 point in it

		Cluster currCluster;

		currCluster.points = points;
		currCluster.centriod = randomPoint;

		return currCluster;

	}

	Point calculateCentriod(Cluster cluster)
	{

		Point newPoint;
		std::array<unsigned int, 3> newRgbAverages;
		std::array<unsigned char, 3> rgb;
		
    long unsigned	int clusterSize {cluster.points.size()};


		//sum all the r g and b values for each point
		for (Point point : cluster.points)
		{
		
			newRgbAverages[0] += static_cast<unsigned int>(point.r);
			newRgbAverages[1] += static_cast<unsigned int>(point.g);
			newRgbAverages[2] += static_cast<unsigned int>(point.b);
		}

		//get averages

		rgb[0] = static_cast<unsigned char>(newRgbAverages[0]/clusterSize);
		rgb[1] = static_cast<unsigned char>(newRgbAverages[1]/clusterSize);
	  rgb[2] = static_cast<unsigned char>(newRgbAverages[2]/clusterSize);
		
		//we now have calculated a new centroid with the new average, so lets assign a point struct with the new data

		newPoint.r = rgb[0];
		newPoint.g = rgb[1];
		newPoint.b = rgb[2];

		return newPoint;
	}

}



