#ifndef OCTTREENODE
#define OCTTREENODE
#include "color.h"
#include <array>
#define CHILDREN 8
#define LEVELS 8
class OctTreeNode {

private:
  std::array<OctTreeNode *, CHILDREN> children;
  int level;
  bool leaf;
  bool reduced;
  int count;
  

public:
  
  Color *color;
  OctTreeNode(int level);
  OctTreeNode(Color *col, int level);
  ~OctTreeNode();
  OctTreeNode *at(int idx);
  std::array<OctTreeNode*, CHILDREN> getChildren();
  void insert(OctTreeNode* node, size_t idx);
  int reduce(int leaves, int target);
  void add(Color *col);
  bool isLeaf();
  bool isReduced();
};

#endif
