#include <iostream>
#include <string>
#include "colors.h"


void printHelp(std::string programName)
{
		std::cout << "Usage:\n " << programName << ": pathToImage numberOfColors\n";
        std::cout << "Example: " << programName << " ~/Pictures/picture.png 8\n";

}

int main(int argc, char** argv)
{


    if (argc != 3){
        printHelp(argv[0]);
        return 1;        

    }


        
	std::string path {argv[1]};//this is our path to the image

	if (path == "--help")
	{
        printHelp(argv[0]);	
		return 0;

	}



	std::string  palletSizeInput {argv[2]};//and this is the size of the color pallet

	int palletSize {std::stoi(palletSizeInput)};	

	std::cout << "Generating a " << palletSize<<" color pallet from " << path << "..." << '\n';		
	
	pallet::makeColorPallet(path, palletSize);

	return 0;

}


