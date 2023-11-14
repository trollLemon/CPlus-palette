#include "CImg.h"
#include <iostream>
#include <string>

#include "CImg.h"
#include "color.h"
#include "k_mean.h"
#include "median_cut.h"
#include "median_cut_helpers.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <unordered_set>
#include <vector>
using namespace cimg_library;


void printResults(std::vector<Color *> &results, std::string &prompt,
                  int limit, std::string fmt) {
  std::cout << prompt << std::endl;

  for (int i = 0; i < limit; ++i) {
    std::cout << results[i]->asHex() << " ";
    
    if (fmt =="RGB"){
       std::cout << ": rgb(" << results[i]->Red() << "," << results[i]->Green() << "," << results[i]->Blue() << ")";
    }
    
    std::cout << std::endl;
  }
}

void makeColorPalette(std::string &path, int size, std::string genType, std::string fmt) {

  CImg<unsigned char> *image = new CImg<unsigned char>(
      path.c_str()); // This is assigned if an image is loaded without errors,
                     // if not , then the program will exit and this wont be
                     // used

  int widthAndHeight{500};
  image->resize(widthAndHeight, widthAndHeight);

  int height{image->height()};
  int width{image->width()};

  std::unordered_set<std::string> seen;
  std::vector<Color *> colors;
  Color base(0, 0, 0);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);

      base.setRGB(r, g, b);
      std::string hex = base.asHex();

      if (seen.count(hex) == 0) {
        colors.push_back(new Color(r, g, b));
        seen.insert(hex);
      }
    }
  }

  if (genType == "-k") {

    std::vector<Color *> palette = KMeans(colors, size);

    std::string prompt = "K Mean Clustering:";
    printResults(palette, prompt,size,fmt);

  } else {
    int depth = powerOfTwoSize(size);
    std::vector<Color *> palette = median_cut(colors, depth);

    std::string prompt = "Median Cut";
    printResults(palette, prompt, size,fmt);
    for (Color *color: palette) {
	delete color;
    }
  }
  for (Color *color : colors) {
    delete color;
  }

  delete image;
}

void printHelp(std::string programName) {
  std::cout << "Usage:\n " << programName
            << ": pathToImage -d [numberOfColors] -t [quantization type] <FORMAT>[' ','RGB'] "
            << std::endl;
  std::cout << "Examples: " << programName << " ~/Pictures/picture.png 8 -k RGB \n          " << programName  << "~/Pictures/picture.png 12 -m \n"
            << std::endl;
  std::cout << "-k: uses K mean Clustering for Color Palette Generation: "
               "slower but produces better palettes most of the time"
            << std::endl;
  std::cout << "-m : used Median Cut for Color Palette Generation: Faster "
               "than K mean Clustering but color palettes aren't always as good"
            << std::endl;
}

int main(int argc, char **argv) {


    std::string path;
    int paletteSize = 8; 
    std::string genType = "-k"; 
    std::string colorFormat = ""; 

    if (argc == 1) {
        printHelp(argv[0]);
        return 1;
    }

    path = argv[1];

    if (argc > 2) {
        // Check if the second argument is a number (palette size) or a generation type
        std::string secondArg = argv[2];
        if (isdigit(secondArg[0])) {
            paletteSize = std::stoi(secondArg);
            if (argc > 3) genType = argv[3];
            if (argc > 4) colorFormat = argv[4];
        } else {
            genType = secondArg;
            if (argc > 3) colorFormat = argv[3];
        }
    } else{
      printHelp(argv[0]);
      return 1;
    }

  try {
    makeColorPalette(path, paletteSize, genType, colorFormat);
  } catch (cimg_library::CImgIOException const &) {
    std::cout << "Failed to load " << path << '\n';
    return 1;
  } catch (cimg_library::CImgArgumentException const &) {
    std::cout << "Failed to load " << path << ", it is a directory" << '\n';
    return 1;
  }
  return 0;
}
