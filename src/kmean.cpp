#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <algorithm>
#include "kmean.h"
#include "dataTypes.h"
double colorDistance(Point& p, Point& q)
	{

    }
  
	void chooseCentroids(std::vector<Cluster>& clusters, std::vector<Point>& points, int k )
	{


		std::vector<int> indecies;
		for (uint i =0; i < points.size(); ++i)
		{	
			indecies.push_back(i);
		}
		std::random_device rd;
		std::seed_seq ss{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() }; 
		std::mt19937 gen{ss};
 
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
		
	}


	void updateCentroids(std::vector<Cluster>& clusters)
	{
        for(Cluster& c: clusters)
        {
            c.calculateNewCentroid();
        }
	}




bool done(std::array<int,3>& a, std::array<int,3>& b)
{
	
	if(abs(a[0]-b[0]) == 0 && abs(a[1]-b[1]) == 0 && abs(a[2]-b[2])==0)
		return true;
	else
		return false;
}




 std::vector<Point> generatePalette(std::vector<std::array<int,3>>& colorData, int size)
 {
		//load image data into points, then put them in the points vector
		std::vector<Point> points;
		std::vector<Cluster> clusters;
	
		for(std::array<int,3>& rgb : colorData)
		{
			Point p {Point(rgb[0], rgb[1], rgb[2])};
			points.push_back(p);
		}
	
		//create clusters and choose some starting centroids
		chooseCentroids(clusters, points, size);
	
		

		
		while(true)
		{
			std::array<int,3> oldRgb {};
		
			for(Cluster& c : clusters)
			{
				oldRgb[0] += c.centroid.r;
				oldRgb[1] += c.centroid.r;
				oldRgb[2] += c.centroid.r;
			}	


			assignPoints(points,clusters);
			updateCentroids(clusters);
				

			std::array<int,3> newRgb {};
		
			for(Cluster& c : clusters)

			{
				newRgb[0] += c.centroid.r;
				newRgb[1] += c.centroid.r;
				newRgb[2] += c.centroid.r;
			}	


			if(done(oldRgb,newRgb))
				 break;
			else
				continue;


		}
		
		std::vector<Point> palette;
		
		for(int i{0}; i< clusters.size(); ++i)
		{
			palette.push_back(clusters.at(i).centroid);
		}		
		return palette;
	}
