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

void printResults(std::vector<std::string> &result, std::string &prompt) {
  std::cout << prompt << std::endl;

  for (std::string color : result) {
    std::cout << color << std::endl;
  }
}

void printResults(std::vector<std::string> &result, std::string &prompt,
                  int limit) {
  std::cout << prompt << std::endl;

  for (int i = 0; i < limit; ++i) {
    std::cout << result[i] << std::endl;
  }
}

void makeColorPalette(std::string &path, int size, int genType) {

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

  if (genType == 1) {

    std::vector<std::string> palette = KMeans(colors, size);

    std::string prompt = "K Mean Clustering:";
    printResults(palette, prompt);

  } else {
    int depth = powerOfTwoSize(size);
    std::vector<std::string> palette = median_cut(colors, depth);

    std::string prompt = "Median Cut";
    printResults(palette, prompt, size);
  }
  for (Color *color : colors) {
    delete color;
  }

  delete image;
}

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

  if (all_args.size() > 2)
    genType = all_args[2];

  if (all_args[0] == "--help") {

    printHelp(argv[0]);
    return 0;
  }
 
  int paletteSize = 8;
  if(all_args.size() >1)
     paletteSize = std::stoi(all_args[1]);

  if (paletteSize <= 0) {
    std::cout << "Cannot make a palette with 0 or less colors" << std::endl;
    return 1;
  }

  if (genType != "-m" && genType != "-k") {
    printHelp(argv[0]);
    return 1;
  }

  int type = 1;
  if (genType == "-m") {
    type = 2;
  }

  try {
    makeColorPalette(path, paletteSize, type);
  } catch (cimg_library::CImgIOException const &) {
    std::cout << "Failed to load " << path << '\n';
    return 1;
  } catch (cimg_library::CImgArgumentException const &) {
    std::cout << "Failed to load " << path << ", it is a directory" << '\n';
    return 1;
  }
  return 0;
}
