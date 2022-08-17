#include <iostream>
#include <string>
#include "colors.h"
int main(int argc, char** argv)
{




	std::string path {argv[1]};//this is our path to the image

	if (path == "--help")
	{
	
		std::cout << "Usage " << argv[0] << ": pathToImage numberOfColors\n";
		return 0;

	}



	std::string  palletSizeInput {argv[2]};//and this is the size of the color pallet

	int palletSize {std::stoi(palletSizeInput)};	

	std::cout << "Generating a " << palletSize<<" color pallet from " << path << "..." << '\n';		
	
	pallet::makeColorPallet(path, palletSize);

	return 0;

}

