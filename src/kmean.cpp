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
					averages[0] += r*r;
					averages[1] += g*g;
					averages[2] += b*b;
					++size;	
			}
				clusters.at(c).getData().clear();
				Point newCentroid = Point(std::sqrt(averages[0]/size), std::sqrt(averages[1]/size), std::sqrt(averages[2]/size));
				clusters.at(c).centroid = newCentroid;
				clusters.at(c).addPoint(newCentroid);

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
