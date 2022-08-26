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
    	indecies.push_back(i);
	
		std::random_device rd;
    std::mt19937 gen{rd()};
 
		std::ranges::shuffle(indecies, gen);
		for(int i{0}; i<k; ++i)
		{

			
		 int randomIndex = indecies.at(i);	
		 std::vector<Point> data;
		 Cluster clstr{Cluster(points.at(randomIndex), i)};
		 clusters.push_back(clstr);
		}
	}
  
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters)
	{
		for (int i{0}; i< points.size(); ++i)
		{
			std::vector<std::tuple<double, int>> distances;
			for(Cluster c: clusters)
			{
				double dist {colorDistance(points.at(i),c.centroid)};
				int theId {c.id};

				distances.push_back(std::make_tuple(dist, theId));

			}

			std::sort(distances.begin(), distances.end());
			
			int closestId {std::get<1>(distances.at(0))};
			
			for (int i{0}; i< clusters.size(); ++i)
			{
				if (clusters.at(i).id == closestId)
				{
					points.at(i).id = closestId;
					clusters.at(i).addPoint(points.at(i));
				}
			}
			
		}		
	}


	void updateCentroids(std::vector<Cluster>& clusters)
	{
		
		std::array<int,3> averages = {0,0,0};
		int size=0;

		for(int c{0}; c < clusters.size(); ++c)
		{
			averages={0,0,0};
			size = 0;
					for(int p{0}; p < clusters.at(c).getData().size(); ++p)
					{
						averages[0] += clusters.at(c).getData().at(p).r * clusters.at(c).getData().at(p).r;
						averages[1] += clusters.at(c).getData().at(p).g * clusters.at(c).getData().at(p).g;
						averages[2] += clusters.at(c).getData().at(p).b * clusters.at(c).getData().at(p).b;
						++size;
					}
				
					Point p {Point(std::sqrt(averages[0]/size), std::sqrt(averages[1]/size), std::sqrt(averages[2]/size))};
					clusters.at(c).setCentroid(p);
					clusters.at(c).data.clear();
					clusters.at(c).getData().push_back(p);
		}
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
	
		
		assignPoints(points, clusters);	
		
		int x = 0;

		while(x<10)
		{
		updateCentroids(clusters);
		assignPoints(points,clusters);
		++x;
		}
		
		std::vector<Point> palette;
		
		for(int i{0}; i< clusters.size(); ++i)
		{
			palette.push_back(clusters.at(i).centroid);
		}		
		return palette;
	}
