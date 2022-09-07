#include <iostream>
#include <string>
#include "colors.h"
#include "CImg.h"

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



	  std::string  palletSizeInput {argv[2]};//and this is the size of the colo:w

	  int palletSize {std::stoi(palletSizeInput)};	

	  std::cout << "Generating a " << palletSize<<" color pallet from " << path << "..." << '\n';		
	
    //take in user inputs and create a color palette, and return an Enum telling us if it 
    //was successful or not
    
    try
    {
    palette::makeColorPalette(path, palletSize);
    }
    catch(cimg_library::CImgIOException)
    {
        std::cout << "Failed to load " << path << '\n';
        return 1;
    }
    catch(cimg_library::CImgArgumentException)
    {
        std::cout << "Failed to load " << path <<", it is a directory" << '\n'; 
        return 1;
    }
	    return 0;

}



