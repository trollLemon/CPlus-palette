#include "loadAndSelect.h"
#include "median_cut_helpers.h"
#include "CImg.h"
#include "color.h"
#include "adv_color.h"
#include "median_cut.h"
#include "k_mean.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <string>
#include <unordered_set>
#include <vector>
#include "k_mean.h"
#include <iostream>
using namespace cimg_library;









void printResults(std::vector<std::string> &result, std::string &prompt ){
    std::cout << prompt << std::endl;

    for (std::string color : result) {
        std::cout <<  color << std::endl;

    }

}


/* *
 * Perform K_Mean clustering to generate a palette
 *
 * */
void DoKMean( CImg<unsigned char> *image, int size ){
  int height{image->height()};
  int width{image->width()};

  std::unordered_set<std::string> seen;
  std::vector<ADV_Color *> colors;
  ADV_Color *base = new ADV_Color(0,0,0);  

  for (int i = 0; i< height; ++i){
        for(int j = 0; j<width; ++j){
           int r =  *image->data(j, i, 0, 0);
           int g =  *image->data(j, i, 0, 1);
           int b =  *image->data(j, i, 0, 2);
         
           base->setRGB(r,g,b);
           std::string hex = base->asHex();

           if (seen.count(hex) == 0) {
               colors.push_back(new ADV_Color(r,g,b)); 
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
        for (ADV_Color *color: colors) {
         delete color;
        }
  

}

/* *
 * Performs median cut to generate a palette
 * */
void DoMedCut(CImg< unsigned char> *image, int size){
  int height{image->height()};
  int width{image->width()};


}




void makeColorPalette(std::string &path, int size, int genType) {

  CImg<unsigned char> *image = new CImg<unsigned char>(
      path.c_str()); // This is assigned if an image is loaded without errors,
                     // if not , then the program will exit and this wont be
                     // used

  int widthAndHeight{500};
  image->resize(widthAndHeight, widthAndHeight);



  if (genType == 1) {
        DoKMean(image, size);
  } else {
        DoMedCut(image, size);
  } 


}


