#include "CImg.h"
#include "loadAndSelect.h"
#include <iostream>
#include <string>
#include <vector>


void printHelp(std::string programName) {
  std::cout << "Usage:\n " << programName
            << ": pathToImage numberOfColors -t [quantization type]"
            << std::endl;
  std::cout << "Example: " << programName << " ~/Pictures/picture.png 8 -k \n"
            << std::endl;
  std::cout << "-k: uses K mean Clustering for Color Palette Generation: "
               "slower but produces better palettes most of the time"
            << std::endl;
  std::cout << "-m : used Median Cut for Color Palette Generation: Faster "
               "than K mean Clustering but color palettes aren't always as good"
            << std::endl;
}

int main(int argc, char **argv) {


std::vector<std::string> all_args;
    
    if (argc == 1) {
        
        printHelp(argv[0]);
        return 1;
    }

    if (argc > 1) 
        all_args.assign(argv + 1, argv + argc);



    std::string path = all_args[0];
    std::string genType = "-k"; 
   
    if(all_args.size() > 2)
        genType = all_args[2];


    if (all_args[0] == "--help") {
        
        printHelp(argv[0]);
        return 0;
    }
    
    int paletteSize = std::stoi(all_args[1]);
        
    if (paletteSize <= 0){
        std::cout << "Cannot make a palette with 0 or less colors" << std::endl;
        return 1;
    }

	
#ifndef USE_CUDA

    if ( genType != "-m" && genType != "-k"  ){
    printHelp(argv[0]);
    return 1;
    }

#else
    if ( genType != "-m" && genType != "-k" && genType != "-c"  ){
    printHelp(argv[0]);
    return 1;
    }

#endif





    int type  = 1;
    if (genType == "-m") {
        type = 2;
    }

   #ifdef USE_CUDA

	if(genType == "-c")
	    type=3;	

   #endif

  try {
    makeColorPalette(path, paletteSize, type);
  } catch (cimg_library::CImgIOException const&) {
    std::cout << "Failed to load " << path << '\n';
    return 1;
  } catch (cimg_library::CImgArgumentException const&) {
    std::cout << "Failed to load " << path << ", it is a directory" << '\n';
    return 1;
  }
  return 0;
}
