#include "oct_tree_node.h"
#include "color.h"
#include <cassert>
#include <iostream>
OctTreeNode::OctTreeNode(int level)
    : color{new Color(0, 0, 0)}, level{level}, leaf{false}, count{0} {
  for (int i = 0; i < CHILDREN; i++) {
    children[i] = nullptr;
  }
}

OctTreeNode::OctTreeNode(Color *col, int level)
    : color{col}, level{level}, leaf(false), count{0} {
  for (int i = 0; i < CHILDREN; i++) {
    children[i] = nullptr;
  }
}

bool OctTreeNode::isLeaf() {

  for (int i = 0; i < CHILDREN; i++) {
    if (children[i] != nullptr)
      return false;
  }

  return true;
}

void OctTreeNode::add(Color *col) {

  count++;
  int red = col->Red() + color->Red();
  int green = col->Green() + color->Green();
  int blue = col->Blue() + color->Blue();

  color->setRGB(red, green, blue);
}


void OctTreeNode::insert(Color *col) {
    OctTreeNode *currentNode = this;
    int depth = 0;

    while (depth < LEVELS) {
        int red = col->Red();
        int green = col->Green();
        int blue = col->Blue();
        int idx = ((red & (1 << (7 - depth))) >> (7 - depth)) << 2 |
                  ((green & (1 << (7 - depth))) >> (7 - depth)) << 1 |
                  ((blue & (1 << (7 - depth))) >> (7 - depth));

	std::cout << idx << std::endl;
        if (currentNode->children[idx] == nullptr) {
            currentNode->children[idx] = new OctTreeNode(depth);
        }

        currentNode = currentNode->children[idx];
        depth++;
    }

    currentNode->add(col);
}


