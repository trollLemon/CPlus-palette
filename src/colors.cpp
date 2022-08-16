#include "colors.h"
#include <string>
#include <map>
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>
#include <bits/stdc++.h>
namespace pallet
{	 

using namespace cimg_library;	 
	

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
	

	std::map<std::array<unsigned char, 3>, long> colors;
	CImg <unsigned char> image(path.c_str());	
	
	image.resize(128,128);

	image.blur_median(20);		
        int height {image.height()};
	int width {image.width()};	
	int increment = 10; //this is how many pixels we incrememnt over	
	
	// CImgDisplay main_disp(image,"Click a point");


	 //while (!main_disp.is_closed()) {
	 
	 //main_disp.wait();

	 //}

	
	for(uint i = 0; i< height; i = i + increment)
	{
		
		
		for(uint j = 0; j < width; j= j + increment)
		{	
			//grab the rgb values of the current pixel
			unsigned char red {image(j,i,0,0)};
			unsigned char green {image(j,i,0,1)};
			unsigned char blue {image(j,i,0,2)};
			
			 std::array<unsigned char, 3> rgb;	
			
			rgb[0] = red;	
			rgb[1] = green;
			rgb[2] = blue;


			//map stuff
			if (!colors.count(rgb))
			{
				colors.insert({rgb, 1});
			} 

			else 
			{
			colors[rgb]++;
			
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
	
	for (int i = 0; i < size; ++i)
	{
	
		std::string currHex;
		long value {colorData.at(i)};
		
		std::array<unsigned char, 3> currRgb;
		
		for(const auto& elem : colors)
		{
			if (elem.second == value)
			{
				currRgb = elem.first;
				break;
			}
		
		}

		colors.erase(currRgb);
		unsigned char red {currRgb[0]};
		unsigned char green {currRgb[1]};
		unsigned char blue {currRgb[2]};

		std::cout << createHex(red,green,blue) << '\n';

	}

	
	}
}

