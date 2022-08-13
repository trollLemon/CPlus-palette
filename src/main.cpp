#include <iostream>
#include <string>
#include "png++/png.hpp"


int main(int argc, char** argv)
{
	
	

	std::string path {argv[1]};//this is our path to the image

	if (path == "--help")
	{
	
		std::cout << "test";
		return 0;

	}



	std::string  palletSizeInput {argv[2]};//and this is the size of the color pallet

	int palletSize {std::stoi(palletSizeInput)};	


	std::cout << "Generating a " << palletSize<<" color pallet from " << path << "..." << '\n';		
	return 0;

}
