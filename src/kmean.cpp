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
		std::random_shuffle(indecies.begin(), indecies.end());

		for(int i{0}; i<k; ++i)
		{


		 int randomIndex = indecies.at(i);	
		 std::cout<< randomIndex << '\n';	
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
		for(int i{0}; i< clusters.size(); i++){

			 clusters.at(i).printCentroid();

			if (clusters.at(i).data.size() == 0)
			{
				break;
			}
			else
			{
			}
		}		

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
      std::cout << "Cluster " << c.id << ": " << c.data.size() << " " << c.centroid.r << " " << c.centroid.g << " " << c.centroid.b << '\n';
    }
		
		std::cout << "\n\n";

		updateCentroids(clusters);
	
		for(Cluster c: clusters)
		{
			std::cout << "Cluster " << c.id << ": " << c.data.size() << " " << c.centroid.r << " " << c.centroid.g << " " << c.centroid.b << '\n';
 		}
	
		std::vector<Point> palette;

		return palette;
	}
