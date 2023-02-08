#include "quantizer.h"
#include <random>
void minHeap::push(double distance) {}
void minHeap::pop() {}
void minHeap::clear() {}

double Quantizer::EuclidianDistance(Color *a, Color *b) { return -99.99; }

void Quantizer::K_MEAN_INIT(int k) {}
void Quantizer::K_MEAN_START() {}

std::vector<std::string> Quantizer::makePalette(std::vector<Color *> &colors,
                                                int k) {

  //load our data into a Red Black tree, in this case a normal map
  //While this data will not be used by the quantizer for the first iteration,
  //the data will every iteration after the first iteration.
  for (Color *point : colors) {
    data[point] = new minHeap();
    colors.push_back(point);
  }

  /* Initialize the Clustering
   *
   *
   *
   * */
  K_MEAN_INIT(k);

  /**/
  K_MEAN_START();

  std::vector<std::string> palette;
  for (int i = 0; i < k; ++i) {

    palette.push_back(clusters[i]->asHex());
    delete clusters[i];
  }

  return palette;
}
