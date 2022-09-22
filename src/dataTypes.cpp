#include "dataTypes.h"
#include <vector>
Point::Point(int red, int green, int blue)
		{
			this->r = red;
			this->g = green;
			this->b = blue;
			this->id = -1; //-1 means the point hasnt been put in a cluster yet
		};
Point::Point()
		{
		this->r =0;
		this->g =0;
		this ->b =0;
		this->id =0;

		};

		int Point::sumRGB()
		{
			return this->r+this->g+this->b;
		};
Cluster::Cluster(Point p, int id)
	{
		this-> centroid =p;
		this-> id = id;
		this-> data.push_back(p);
	};

	void Cluster::addPoint(Point p){
		this->data.push_back(p);
	};
	void Cluster::setCentroid(Point p){
		
		this->centroid = p;
		
		};
	std::vector<Point>& Cluster::getData(){
		return this->data;
	};
	void Cluster::resetCentroid(){
		this->data.clear();
	};

		



	

