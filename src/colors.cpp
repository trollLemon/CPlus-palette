#include "colors.h"
#include "kmean.h"

#include <string>
#include <map>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>

namespace pallet
{	 

using namespace cimg_library;	 
	

	//convert rgb values to hexidecimal
	//gotten from lindevs: https://www.youtube.com/watch?v=TXMegco45q8
	 std::string createHex(int r, int g, int b)
	{
		
		char hex[8];
		std::snprintf( hex, sizeof hex, "#%02x%02x%02x", r,g,b);
		
		std::string hexString;	
	
		for (char i : hex)
		{
			hexString += i;
		
		}	

		return hexString;	
	
	}

	
	/* This is what generates our color pallet. 
	 * We are going to go through every pixel and use a hashmap to keep
	 * track of the frequency of the colors as we loop through each pixel.
	 * Once we have a hashmap with colors and their frequencies, we will 
	 * sort it from most prominent color to least prominent, and then grab
	 * the amount of colors dictated by the size variable.
	 */ 
	void makeColorPallet(std::string path, int size)
	{
	

	
	CImg <unsigned char> image(path.c_str());	
	
	int widthAndHeight{128};
	int blurFactor{10};
	image.resize(widthAndHeight,widthAndHeight);
	image.blur_median(blurFactor);  
	int height {image.height()};
	int width {image.width()};	
	
	//get the colors from each pixel
	
	std::vector<std::array<unsigned char, 3>> colors;

	for(int h{0}; h < height; ++h)
	{
		for(int w{0}; w < width; ++w)
		{
			std::array<unsigned char, 3> pixelColor;
			pixelColor[0] = image(h,w,0,0);
			pixelColor[1] = image(h,w,0,1);
			pixelColor[2] = image(h,w,0,2);
			
			colors.push_back(pixelColor);
		}
			

	}
	kmean_cluster::makePalette (colors);


	}
}
