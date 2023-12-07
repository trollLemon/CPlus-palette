#ifndef OCTTREENODE
#define OCTTREENODE
#include "color.h"
#include <array>
#define CHILDREN 8
#define LEVELS 8
class OctTreeNode {

private:
  std::array<OctTreeNode *, CHILDREN> children;
  Color *color;
  int level;
  bool leaf;
  int count;
  void add(Color *col);
  

public:
  OctTreeNode(int level);
  OctTreeNode(Color *col, int level);
  ~OctTreeNode();
  void insert(Color *col);
  void reduce();
  bool isLeaf();
};

#endif
