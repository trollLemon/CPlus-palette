#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"
#include "stb_image_resize2.h"

#include "color.h"
#include "k_mean.h"
#include "median_cut.h"
#include "median_cut_helpers.h"
#include <cstddef>
#include <iostream>
#include <unordered_set>
#include <vector>

void printResults(std::vector<Color *> &results, std::string &prompt, int limit,
                  std::string fmt) {
  std::cout << prompt << std::endl;

  for (int i = 0; i < limit; ++i) {
    std::cout << results[i]->asHex() << " ";

    if (fmt == "RGB") {
      std::cout << ": rgb(" << results[i]->Red() << "," << results[i]->Green()
                << "," << results[i]->Blue() << ")";
    }

    std::cout << std::endl;
  }
}

void makeColorPalette(std::string &path, int size, std::string genType,
                      std::string fmt) {

  int widthHeight = 500;
  std::vector<Color *> colors;

  int width, height, channels;
  unsigned char *image = stbi_load(path.c_str(), &width, &height, &channels, 0);

  int stride = width * channels;

  std::unordered_set<std::string> seen;
	
 

  if (image == NULL) {
    std::cout << "Unable to load image" << std::endl;
    exit(1);
  }

  unsigned char *resizedImage = stbir_resize_uint8_srgb(
      image, width, height, stride, NULL, widthHeight, widthHeight, stride, STBIR_RGB);
  Color base(0, 0, 0);

for (int y = 0; y < widthHeight; ++y) {
    for (int x = 0; x < widthHeight; ++x) {
    int i = (y * width + x) * channels;
    unsigned char r = resizedImage[i];
    unsigned char g = resizedImage[i+1];
    unsigned char b = resizedImage[i+2];

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
    printResults(palette, prompt, size, fmt);

  } else {
    int depth = powerOfTwoSize(size);
    std::vector<Color *> palette = median_cut(colors, depth);

    std::string prompt = "Median Cut";
    printResults(palette, prompt, size, fmt);
    for (Color *color : palette) {
      delete color;
    }
  }
  for (Color *color : colors) {
    delete color;
  }
  stbi_image_free(image);
  stbi_image_free(resizedImage);
}

void printHelp(std::string programName) {
  std::cout << "Usage:\n " << programName
            << ": pathToImage -d [numberOfColors] -t [quantization type] "
               "<FORMAT>[' ','RGB'] "
            << std::endl;
  std::cout << "Examples: " << programName
            << " ~/Pictures/picture.png 8 -k RGB \n          " << programName
            << "~/Pictures/picture.png 12 -m \n"
            << std::endl;
  std::cout << "quantization types:" << std::endl;
  std::cout << "-k: uses K mean Clustering for Color Palette Generation: "
               "slower but produces better palettes most of the time"
            << std::endl;
  std::cout << "-m : used Median Cut for Color Palette Generation: Faster "
               "than K mean Clustering but color palettes aren't always as good"
            << std::endl;
  std::cout << "FORMAT types:" << std::endl;
  std::cout << "' ' leave empty for hex color codes" << std::endl;
  std::cout
      << "\"RGB\" for additional RGB color values along with the hex colors"
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
    // Check if the second argument is a number (palette size) or a generation
    // type
    std::string secondArg = argv[2];
    if (isdigit(secondArg[0])) {
      paletteSize = std::stoi(secondArg);
      if (argc > 3)
        genType = argv[3];
      if (argc > 4)
        colorFormat = argv[4];
    } else {
      genType = secondArg;
      if (argc > 3)
        colorFormat = argv[3];
    }
  } else {
    printHelp(argv[0]);
    return 1;
  }

  makeColorPalette(path, paletteSize, genType, colorFormat);
  return 0;
}
