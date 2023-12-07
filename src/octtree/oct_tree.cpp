#include "oct_tree.h"
#include "oct_tree_node.h"
#include <vector>

std::vector<Color *> oct_tree_gen(std::vector<Color *> colors, int size) {

  std::vector<Color *> palette;

  OctTreeNode *root = new OctTreeNode(0);

  for (int i = 0; i < colors.size(); i++) {
    root->insert(colors[i]);
  }

  return palette;
}
