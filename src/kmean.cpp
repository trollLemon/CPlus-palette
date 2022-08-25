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
		
		for(int c{0}; c < clusters.size(); ++c)
		{
			
			//dont operate on the centroid if it only consists of a centroid and no data
			if(clusters.at(c).getData().size() ==0)
			{
				continue;
			}

			int aveR{0};
			int aveG{0};
			int aveB{0};
			int size{0};//keep track of the size of the clusters' point vector
			
			for(int p{0}; p < clusters.at(c).getData().size(); ++p)
			{
				//add each r g and b values to the average color variables defined above
				//we get better color averages if we collect an average of the squares
				aveR += clusters.at(c).getData().at(p).r * clusters.at(c).getData().at(p).r;					
				aveG += clusters.at(c).getData().at(p).g * clusters.at(c).getData().at(p).g;
				aveB += clusters.at(c).getData().at(p).b * clusters.at(c).getData().at(p).b;
				++size;
			}

			
			int r {static_cast<int>(std::sqrt(aveR/size))};
			int g {static_cast<int>(std::sqrt(aveG/size))};
			int b {static_cast<int>(std::sqrt(aveB/size))};
		
			Point newCentroid {Point(r,g,b)};
			clusters.at(c).setCentroid(newCentroid);
			clusters.at(c).resetCentroid();	
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

		std::cout << "\n\n";
		
		int x = 0;

		while(x<8)
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
