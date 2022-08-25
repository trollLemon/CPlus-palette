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
		int begining {0};
		std::mt19937 mt {std::random_device{}() };
		std::uniform_int_distribution index{ begining, (int)points.size() };

		for(int i{0}; i<k; ++i)
		{


		 int randomIndex = index(mt);	
			std::vector<Point> data;
		 Cluster clstr{
			points.at(randomIndex),
			data,	
			i
		 };
		 clusters.push_back(clstr);

		}
	}
  
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters)
	{
		//TODO:change this to a normal for loop cause for each loops copy shit and we have 2 bytes of memory	
		for (Point p : points)
		{
	
			std::vector<std::tuple<double, int>> distances;
			for(Cluster c: clusters)
			{
				double dist {colorDistance(p,c.centroid)};
				int theId {c.id};

				distances.push_back(std::make_tuple(dist, theId));

			}

			std::sort(distances.begin(), distances.end());
			
			int closestId {std::get<1>(distances.at(0))};
			
			for (int i{0}; i< clusters.size(); ++i)
			{
				if (clusters.at(i).id == closestId)
				{
					p.id = closestId;
					clusters.at(i).addPoint(p);
				}
			}
			
		}		
	}


	void updateCentroids(Cluster& cluster)
	{
		
	}

  std::vector<Point> generatePalette(std::vector<std::array<int, 3>> colorData, int size)
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
	
		for(Cluster c: clusters)
		{
			std::cout << "Cluster " << c.id << ": " << c.data.size() << '\n';
		}
	
		std::vector<Point> palette;

		return palette;
	}
