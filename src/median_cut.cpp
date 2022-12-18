#include "median_cut.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>
// custom compare functions for the median cut algorithm
bool cmpR(Color *a, Color *b) { return a->Red() > b->Red(); }

bool cmpG(Color *a, Color *b) { return a->Green() > b->Green(); }
bool cmpB(Color *a, Color *b) { return a->Blue() > b->Blue(); }

int getRangeR(std::vector<Color *> colors) {

    int range = 0;

    auto max = std::max_element(colors.begin(), colors.end(), cmpR);

    auto min = std::min_element(colors.begin(), colors.end(), cmpR);

    range = (*max)->Red() - (*min)->Red();

    return std::abs(range);
}
int getRangeG(std::vector<Color *> colors) {

    int range = 0;

    auto max = std::max_element(colors.begin(), colors.end(), cmpG);

    auto min = std::min_element(colors.begin(), colors.end(), cmpG);

    range = (*max)->Green() - (*min)->Green();
    return std::abs(range);
}
int getRangeB(std::vector<Color *> colors) {

    int range = 0;

    auto max = std::max_element(colors.begin(), colors.end(), cmpB);

    auto min = std::min_element(colors.begin(), colors.end(), cmpB);

    range = (*max)->Blue() - (*min)->Blue();
    return std::abs(range);
}

// recursive method based on a python implementation:
// https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
void MedianCut::median_cut(std::vector<Color *> colors, int k) {
    

    if (colors.size() == 0) {
        return;
    }

    if (k == 0) {
        getAverageColor(colors);
        return;
    }

    int rangeR = getRangeR(colors);
    int rangeG = getRangeG(colors);
    int rangeB = getRangeB(colors);


    if (rangeR > rangeG && rangeR > rangeB){
        std::sort(colors.begin(), colors.end(), cmpR);
    }
    if (rangeG > rangeR && rangeG > rangeR){
        std::sort(colors.begin(), colors.end(), cmpG);
    }
    if (rangeB > rangeG && rangeB > rangeR){
        std::sort(colors.begin(), colors.end(), cmpB);
    }
    //divide vector into two new ones and recurse
    int median = colors.size()/2;
    std::vector<Color *> low(colors.begin(), colors.begin()+median);
    std::vector<Color *> high(colors.begin()+median, colors.end());
    median_cut(low, k-1);
    median_cut(high, k-1);

}

void MedianCut::getAverageColor(std::vector<Color *> colors) {

    int aveR = 0;
    int aveG = 0;
    int aveB = 0;
    int size = colors.size();

    for (Color *c : colors) {
        aveR += c->r;
        aveG += c->g;
        aveB += c->b;
    }

    Color *aveColor = new Color(aveR / size, aveG / size, aveB / size);
    std::string aveAsHex = aveColor->asHex();
    delete aveColor;
    this->colors.push_back(aveAsHex);
}

std::vector<std::string> MedianCut::makePalette(std::vector<Color *> colors,
                                                int k) {

    median_cut(colors, k);
    for(auto i : this->colors){
        std::cout <<i << std::endl;
    }
    std::cout << this->colors.size() << std::endl;
    return this->colors;
}

