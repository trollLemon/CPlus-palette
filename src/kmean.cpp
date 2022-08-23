#include <vector>
#include <array>
#include <cmath>
#include "kmean.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <tuple>
#include <bits/stdc++.h>
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
		currCluster.centroid = randomPoint;

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


	double distance(Point p, Point q)
	{
		float r = static_cast<float>(p.r) - static_cast<float>(q.r);
		float g = static_cast<float>(p.g) - static_cast<float>(q.g);
		float b = static_cast<float>(p.b) - static_cast<float>(q.b);
		
		int power = 2;
		return std::sqrt( std::pow(r, power) + std::pow(g, power) + std::pow(b, power));
	}

	 std::array<double,3> centriodDifference(Cluster a, Cluster b)
	 {
			double rDiff {static_cast<double>(a.centroid.r) - static_cast<double>(b.centroid.r)};
			double gDiff {static_cast<double>(a.centroid.g) - static_cast<double>(b.centroid.g)};
		  double bDiff {static_cast<double>(a.centroid.b) - static_cast<double>(b.centroid.b)};

			std::array<double, 3> theDifferance;
			
			theDifferance[0] = std::abs(rDiff);
			theDifferance[1] = std::abs(gDiff);
			theDifferance[2] = std::abs(bDiff);

			return theDifferance;

	 }

	
	std::vector<Point> makePointsFromImageData (std::vector<std::array<unsigned char, 3>> colorData)
	{
		std::vector<Point> thePoints;
		for (auto& rgb : colorData)
		{
			Point newPoint;
			newPoint.r = rgb[0];
			newPoint.g = rgb[1];
			newPoint.b = rgb[2];

			thePoints.push_back(newPoint);
		}

		return thePoints;

	}
	
	bool sortBySecond(std::tuple<Cluster, double> a, std::tuple<Cluster, double> b)
	{

		return (std::get<1>(a) < std::get<1>(b));

	}

	std::vector<Point> makePalette (std::vector<std::array<unsigned char ,3>> colorData, int size)
	{

		std::vector<Point> palette; //this will end up holding our centroids, since those will be the color palette once the k means is done
		
		std::vector<Point> pointData {makePointsFromImageData(colorData)};
		
		//pick 'size' amount of Points randomly and make them into clusters:
		std::vector<Cluster> clusters; //our clusters

		std::mt19937 mt{ std::random_device{}() }; 
    std::uniform_int_distribution randomPoints{ 0, static_cast<int>(pointData.size()) };
		//assign random points to be our cluster centroids
		for(int i{0}; i < size; ++i)
		{
			Cluster currCluster {makeCluster(pointData.at(randomPoints(mt)))};
			clusters.push_back(currCluster);
		}

		/* Now that we have initialized some clusters, we can now start the K means algorithm.
		 * We will add each point to the closest clusture (based on color distance), and 
		 * calculate new centroids for the clusters. 
		 *
		 * We will do this in a while loop, and break once all the clusters have the following observation:
		 * The difference from the old centroid and the new centroid is less than or equal to maxDiff, defined in kmeans.h
		 *
		 * Once we have done that, we can simply fill the palette vector with our cluster centroids, since those will be 
		 * the colors in the palette.
		 */
		
			while (true) 
			{

				//store the centroids of each cluster before reassignment, we will compare these later
				std::vector<Point> oldCentroids;

				for (Cluster c: clusters)
				{
					oldCentroids.push_back(c.centroid);
				}
				
				//assign points to clusters
				for (Point p : pointData)
				{
					std::vector<std::tuple<Cluster, double>> distances;
					
					//get the distance from the current point to each centroid in the clusters.
					//we group the distance and the centroid in a tuple, then add it to a vector 
					for (Cluster c : clusters)
					{
					double theDistance {distance(p, c.centroid)};
					distances.push_back(std::make_tuple(c, theDistance));
				
					}

					//now find the closest cluster and add the point to the cluster.

					std::sort(distances.begin(), distances.end(), sortBySecond);

				}	





			}	


		return palette;

	}	
}



