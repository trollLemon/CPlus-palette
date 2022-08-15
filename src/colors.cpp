#include "colors.h"
#include "png++/png.hpp"
#include <string>
#include <map>
#include<vector>
#include <algorithm>
#include <sstream>
#include <bits/stdc++.h>
namespace pallet
{	 

	//return the pixel at the given indexes
 	 png::rgb_pixel  getPixel ( png::image< png::rgb_pixel > image, int width, int height){
		 return image[height][width];
	 
	 }
	

	//convert rgb values to hexidecimal
	// code snippet by Nikos C from this stackoverflow post: https://stackoverflow.com/questions/14375156/how-to-convert-a-rgb-color-value-to-an-hexadecimal-value-in-c 
	 std::string createHex(int r, int g, int b)
	{
	
		std::stringstream hex;

		hex << "#";
		hex << std::hex << (r << 16 | g << 8 | b);
		return hex.str();	
	
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
	

	std::map<std::string, long> colors;
	
		
	png::image< png::rgb_pixel > image(path);//read the image
	image.resize((image.get_width()* 0.1),(image.get_height()*0.1));//resize image
        uint height {image.get_height()};
	uint width {image.get_width()};	
	uint increment = 10; //this is how many pixels we incrememnt over	
	
	for(uint i = 0; i< height; i = i + increment)
	{
		
		
		for(uint j = 0; j < width; j= j + increment)
		{
			png::rgb_pixel pixel {getPixel(image, j, i)};

			std::string hex { createHex(pixel.red, pixel.green, pixel.blue)};
			//map stuff
			if (!colors.count(hex))
			{
				colors.insert({hex, 1});
			} 

			else 
			{
			colors[hex]++;
			
			}
		}	
	}
	
	std::cout << "Got color data, making pallet \n";	
	//now that we have the colors and their frequencies, we put them in an array list and sort it
	std::vector<long> colorData;	
	for(const auto& elem : colors)
	{
		colorData.push_back(elem.second);
	}	   
	
	std::sort(colorData.begin(), colorData.end());	
	std::reverse(colorData.begin(), colorData.end());	
        //now print the color pallet to the user
	
	std::cout << "Color Pallet:" << '\n';

	for (int i = 0; i < size; i++ )	
	{
		std::string currHex ;	
		long value {colorData.at(i)};

		//find the key corresponding to the value
		for (const auto& elem : colors)
		{
			if(elem.second == value)
			{
			    
			    currHex = elem.first;
			}	
		}
		
		std::cout << currHex << '\n';
	}	

	}

}

