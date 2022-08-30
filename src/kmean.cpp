#include "kmean.h"
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <algorithm>
  double colorDistance(Point& p, Point& q)
	{
		int diffR {(p.r-q.r) * (p.r-q.r)};
		int diffG {(p.g-q.g)*(p.g-q.g)};
		int diffB {(p.b-q.b) * (p.b-q.b)};
		
		return std::sqrt(diffR + diffG + diffB);
	}
  
	void chooseCentroids(std::vector<Cluster>& clusters, std::vector<Point>& points, int k )
	{

		std::vector<int> indecies;
		for (int i {0}; i < points.size(); ++i)
		{	
			indecies.push_back(i);
		}
		std::random_device rd;
    std::mt19937 gen{rd()};
 
		std::ranges::shuffle(indecies, gen);
		for(int i{0}; i<k; ++i)
		{	
		 int randomIndex = indecies.at(i);	
		 Cluster clstr{Cluster(points.at(randomIndex), i)};
		 clusters.push_back(clstr);
		}
	}
  
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters)
	{
		

		for(int p {0}; p < points.size(); ++p)
		{

			//find closest cluster
			
			int closest = clusters.at(0).id;
			double closestDist = colorDistance(points.at(p), clusters.at(0).centroid);
			for(int c {0}; c < clusters.size(); ++c)
			{
				double currDist = colorDistance(points.at(p), clusters.at(c).centroid);

				if(currDist < closestDist )
				{
					closestDist = currDist;
					closest = clusters.at(c).id;
				}
			}
			
			//add the point to the closest cluster
		
			for (int c{0}; c< clusters.size(); ++c)
			{
				if (clusters.at(c).id == closest)
				{
					clusters.at(c).addPoint(points.at(p));
					break;
				}
			}

		}

	}


	void updateCentroids(std::vector<Cluster>& clusters)
	{
		for (int c{0}; c < clusters.size(); ++c)
		{
			std::array<int,3> averages = {0,0,0};
			int size = 0;
			for(int p{0}; p < clusters.at(c).getData().size(); ++p)
			{
					int r = clusters.at(c).getData().at(p).r;
					int g = clusters.at(c).getData().at(p).g;
					int b = clusters.at(c).getData().at(p).b;
					averages[0] += r;
					averages[1] += g;
					averages[2] += b;
					++size;	
			}

				Point g = Point(averages[0]/size, averages[1]/size, averages[2]/size);
				clusters.at(c).centroid = g;			
		}
	}

	//determine if the centroids have converged
	bool shouldContinue(std::vector<Point> oldCentroids, std::vector<Point> newCentroids)
	{
		
		std::vector<std::array<int,3>> diffs;

		for (int i {0}; i< oldCentroids.size(); /*both vectors will be the same size, so we can set the limit to any of them*/ ++i)
			{
			
				Point a {oldCentroids.at(i)};
				Point b {newCentroids.at(i)};

				std::array<int,3> currRgb {a.r-b.r,a.g-b.g,a.b-b.b};//get differences of the two sets of RGB values
				diffs.push_back(currRgb);
			}

		//if any of the differences are greater than 0, then we should still go through the main loop
		for (std::array<int,3> rgb : diffs)
		{
			if(rgb[0]> 0 || rgb[1] > 0 || rgb[2] > 0)
				return true;
		}

		return false;
	}

 std::vector<Point> generatePalette(std::vector<std::array<int,3>> colorData, int size)
 {
		//load image data into points, then put them in the points vector
		std::vector<Point> points;
		std::vector<Cluster> clusters;
	
		for(std::array<int,3> rgb : colorData)
		{
			Point p {Point(rgb[0], rgb[1], rgb[2])};
			points.push_back(p);
		}
	
		//create clusters and choose some starting centroids
		chooseCentroids(clusters, points, size);
	
		
		
		

		while(true)
		{
			std::vector<Point> oldCentroids;
			//store old centroid data
		
			for(Cluster c : clusters)
			{
				oldCentroids.push_back(c.centroid);
			}	


			assignPoints(points,clusters);
			updateCentroids(clusters);
		
			//store new centroids
			std::vector<Point> newCentroids;

			for (Cluster c : clusters)
			{
				newCentroids.push_back(c.centroid);
			}

			//see if we should keep looping
			if(shouldContinue(oldCentroids, newCentroids))
				continue;
			else
				break;

		}
		
		std::vector<Point> palette;
		
		for(int i{0}; i< clusters.size(); ++i)
		{
			palette.push_back(clusters.at(i).centroid);
		}		
		return palette;
	}
