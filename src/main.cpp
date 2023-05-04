#include "CImg.h"
#include "loadAndSelect.h"
#include <iostream>
#include <string>
void printHelp(std::string programName) {
  std::cout << "Usage:\n " << programName
            << ": pathToImage numberOfColors -t [quantization type]"
            << std::endl;
  std::cout << "Example: " << programName << " ~/Pictures/picture.png 8 -t 1\n"
            << std::endl;
  std::cout << "-t 1: uses K mean Clustering for Color Palette Generation: "
               "slower but produces better palettes most of the time"
            << std::endl;
  std::cout << "-t 2: used Median Cut for Color Palette Generation: Faster "
               "than K mean Clustering but color palettes aren't always as good"
            << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printHelp(argv[0]);
    return 1;
  }

  std::string path{argv[1]}; // this is our path to the image

  if (path == "--help") {
    printHelp(argv[0]);
    return 0;
  }

  std::string paletteSizeInput{argv[2]}; // and this is the size of the color

  int paletteSize{std::stoi(paletteSizeInput)};

  if (paletteSize <= 0) {
    std::cout << "Cannot make a palette with size less or equal to 0"
              << std::endl;
    return 0;
  }

  std::cout << "Generating a " << paletteSize << " color palette from " << path
            << "..." << '\n';

  int genType = 1;

  if(argc > 4){
  if (argv[4]) {
    genType = *argv[4] - '0';
  }


  if (genType < -1 || genType > 2) {
    printHelp(argv[0]);
    return 1;
  }

  }
  // take in user inputs and create a color palette, and return an Enum
  // telling us if it was successful or not

  try {
    makeColorPalette(path, paletteSize, genType);
  } catch (cimg_library::CImgIOException) {
    std::cout << "Failed to load " << path << '\n';
    return 1;
  } catch (cimg_library::CImgArgumentException) {
    std::cout << "Failed to load " << path << ", it is a directory" << '\n';
    return 1;
  }
  return 0;
}
