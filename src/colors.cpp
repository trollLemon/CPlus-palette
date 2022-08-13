#include "colors.h"
#include "png++/png.hpp"
#include <string>



namespace pallet
{	 

	//return the pixel at the given indexes
 	 png::rgb_pixel  getPixel ( png::image< png::rgb_pixel > image, int width, int height){
	 
		 return image[width][height];
	 
	 }
	

	//convert rgb values to hexidecimal
	// code snippet by Nikos C from this stackoverflow post: https://stackoverflow.com/questions/14375156/how-to-convert-a-rgb-color-value-to-an-hexadecimal-value-in-c 
	unsigned long createRGB(int r, int g, int b)
	{
	 return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
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
	
	png::image< png::rgb_pixel > image(path);//read the image

	uint height {image.get_height()};
	uint width {image.get_width()};

	for (uint h; h < height; ++h)
	{
		for(uint w; w < width; ++w )
		{
		
		}
	}


	}


}

