#include "colors.h"
#include <string>
#include <map>
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>
#include <bits/stdc++.h>
#include <tuple>
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
	

	
	CImg <unsigned char> image(path.c_str());	
	
	image.resize(128,128);

	image.blur_median(10);		
        int height {image.height()};
	int width {image.width()};	
	
  CImgDisplay main_disp(image,"Click a point");


	 while (!main_disp.is_closed()) {
	 
	 main_disp.wait();

	 }
	//split the resized image into blocks, and find the average pixel value of each block
	std::vector<std::array<unsigned char, 3>> averageColors;
	for (int i{0}; i<size; ++i)
	{
		int increment = 128/size; //increment for the block loop, this increases by 128/size each iteration	
		int w{increment * i};
		int h{increment *i};
	
		 std::vector<std::array<unsigned char, 3>> colors;
		
		for(; w < increment * (i + 1) &&  h < increment * (i + 1); ++w, ++h)
		{
			
			
			unsigned char red {image(h,w,0,0)};
			unsigned char green {image(h,w,0,1)};
			unsigned char blue {image(h,w,0,2)};
												
			std::array<unsigned char, 3> rgb;	
														 			
			rgb[0] = red;	
			rgb[1] = green;
			rgb[2] = blue;
			
			colors.push_back(rgb);	
		}
		
		//get average
		
		int averageRed;
		int averageGreen;
		int averageBlue;
		int sampleSize {3};
		for (auto& rgbValues : colors)
		{
			averageRed +=(int) rgbValues[0];
			averageGreen += (int) rgbValues[1];
			averageBlue += (int)rgbValues[2];
		
		
		}	

		std::array<unsigned char, 3> average;
		
		average[0] = averageRed/sampleSize;
		average[1] = averageGreen/sampleSize;
		average[2] = averageBlue/sampleSize;
		
		averageColors.push_back(average);

	}	

	
	std::cout << "Got color data, making pallet \n";	
	
	//convert rgb to hex and print to console
	
	for (auto& rgb : averageColors)
	{
		std::string hex {createHex(rgb[0], rgb[1],rgb[2])};
		std::cout << hex << '\n';
	
	}	


	}	
		

	

	
	
}

