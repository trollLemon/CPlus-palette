#include "oct_tree_node.h"
#include "color.h"
#include <array>
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

OctTreeNode::~OctTreeNode(){}


OctTreeNode *OctTreeNode::at(int idx) { return children[idx]; };

bool OctTreeNode::isLeaf() {

  return leaf;
}

std::array<OctTreeNode *, CHILDREN> OctTreeNode::getChildren() {
  return children;
}

void OctTreeNode::reduce(){

	int ave_r =0;
	int ave_g=0;
	int ave_b =0;
        int count =0;
	for(OctTreeNode *child: children){
		if(child==nullptr) continue;
		Color * col = child->color;
		ave_r+=col->Red();
		ave_g+=col->Green();
		ave_b+=col->Blue();
		count++;
		leaf =true;
	}

	color->setRGB(ave_r/count, ave_g/count, ave_b/count);

}

void OctTreeNode::add(Color *col) {
  count++;
  int red = col->Red() + color->Red();
  int green = col->Green() + color->Green();
  int blue = col->Blue() + color->Blue();
  red/=count;
  green/=count;
  blue/=count;
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

    if (currentNode->children[idx] == nullptr) {
      currentNode->children[idx] = new OctTreeNode(depth);
    }

    currentNode = currentNode->children[idx];
    depth++;
  }
  
  currentNode->leaf = true;
  currentNode->add(col);
}
