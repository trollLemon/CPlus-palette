#include "loadAndSelect.h"
#include "CImg.h"
#include "adv_color.h"
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
#include <string>
#include <unordered_set>
#include <vector>

#ifdef USE_CUDA
#include "k_mean_cuda.h"

#endif

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

/* *
 * Perform K_Mean clustering to generate a palette
 *
 * */
void DoKMean(CImg<unsigned char> *image, int size) {
  int height{image->height()};
  int width{image->width()};


  std::unordered_set<std::string> seen;
  std::vector<ADV_Color *> colors;
  ADV_Color *base = new ADV_Color(0, 0, 0);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);

      base->setRGB(r, g, b);
      std::string hex = base->asHex();

      if (seen.count(hex) == 0) {
        colors.push_back(new ADV_Color(r, g, b));
        seen.insert(hex);
      }
    }
  }

  KMean *proc = new KMean();
  std::vector<std::string> palette = proc->makePalette(colors, size);

  std::string prompt = "K Mean Clustering:";
  printResults(palette, prompt);

  delete proc;
  delete base;
  delete image;
  for (ADV_Color *color : colors) {
    delete color;
  }
}

/* *
 * Performs median cut to generate a palette
 * */
void DoMedCut(CImg<unsigned char> *image, int size) {

  int height{image->height()};
  int width{image->width()};

  std::unordered_set<std::string> seen;
  std::vector<Color *> colors;
  Color *base = new Color(0, 0, 0);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);

      base->setRGB(r, g, b);
      std::string hex = base->asHex();

      if (seen.count(hex) == 0) {
        colors.push_back(new Color(r, g, b));
        seen.insert(hex);
      }
    }
  }
  MedianCut *proc = new MedianCut();
  int depth = powerOfTwoSize(size);
  std::vector<std::string> palette = proc->makePalette(colors, depth);

  std::string prompt = "Median Cut";
  printResults(palette, prompt, size);

  delete proc;
  delete base;
  delete image;
  for (Color *color : colors) {
    delete color;
  }
}

void makeColorPalette(std::string &path, int size, int genType) {

  CImg<unsigned char> *image = new CImg<unsigned char>(
      path.c_str()); // This is assigned if an image is loaded without errors,
                     // if not , then the program will exit and this wont be
                     // used

  if (genType == 1) {
    int widthAndHeight{500};
    image->resize(widthAndHeight, widthAndHeight);

    DoKMean(image, size);
  }
#ifdef USE_CUDA
  else if (genType == 3) {
  int widthAndHeight{500};
  image->resize(widthAndHeight, widthAndHeight);
  int height{image->height()};
  int width{image->width()};
  int psize = height * width;
	pixel* pixelArray = new pixel[psize];
 int pixelIndex = 0; 
  std::unordered_set<std::string> seen;
  Color helper = Color(0,0,0);
    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int r = *image->data(j, i, 0, 0);
      int g = *image->data(j, i, 0, 1);
      int b = *image->data(j, i, 0, 2);
      helper.setRGB(r,g,b);      
      if(seen.count(helper.asHex())==1) continue;	

            pixelArray[pixelIndex].r = r;
            pixelArray[pixelIndex].g = g;
            pixelArray[pixelIndex].b = b;
            pixelIndex++;
	    seen.insert(helper.asHex());
      }
    }


    std::vector<std::string> palette = CudaKmeanWrapper(pixelArray, size, pixelIndex);
    std::string prompt = "K-mean clustering GPU accelerated:::";
    printResults(palette, prompt);
    delete[] pixelArray;
  }
#endif

  else {
    int widthAndHeight{500};
    image->resize(widthAndHeight, widthAndHeight);

    DoMedCut(image, size);
  }
}
