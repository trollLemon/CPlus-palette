#ifndef KMEAN
#define KMEAN
#include <iostream>
#include <vector>
#include <array>
	struct Point {
	int r;
	int g;
	int b;

	int id;
		Point(int red, int green, int blue)
		{
			this->r = red;
			this->g = green;
			this->b = blue;
			this->id = -1; //-1 means the point hasnt been put in a cluster yet
		};
		Point()
		{
		this->r =0;
		this->g =0;
		this ->b =0;
		this->id =0;

		};
		void printRgb()
		{

			std::cout << this->r << " " << this->g << " " << this->b << " " << '\n'; 

		};

		int sumRGB()
		{
			return this->r+this->g+this->b;
		};

	};
	
	
	struct Cluster {
  
  Point centroid;
	std::vector<Point> data;
  int id;
	
	Cluster(Point p, int id)
	{
		this-> centroid =p;
		this-> id = id;
		this-> data.push_back(p);
	};

	void addPoint(Point p){
		this->data.push_back(p);
	};
  void printCentroid(){

		this->centroid.printRgb();
  };
	void setCentroid(Point p){
		
		this->centroid = p;
		
		};
	std::vector<Point>& getData(){
		return this->data;
	};
	void resetCentroid(){
		this->data.clear();
	};

		
	};


	
	double colorDistance(Point& p, Point& q);
	void chooseCentroids(std::vector<Cluster>& clusters, std::vector<Point>& points, int k );
	void assignPoints(std::vector<Point>& points, std::vector<Cluster>& clusters);
	void updateCentroids(std::vector<Cluster>& clusters);
	
	std::vector<Point> generatePalette(std::vector<std::array<int,3>>& colorData, int size);

#endif
