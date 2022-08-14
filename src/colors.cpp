#include "colors.h"
#include "png++/png.hpp"
#include <string>
#include <map>


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
        uint height {image.get_height()};
	uint width {image.get_width()};	
	uint increment = 100; //this is how many pixels we incrememnt over	
	
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
	
		
	}

}

