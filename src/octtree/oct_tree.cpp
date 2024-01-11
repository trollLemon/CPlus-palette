#include "oct_tree.h"
#include "color.h"
#include "oct_tree_node.h"
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

static std::vector<OctTreeNode *> get_leaf_nodes(OctTreeNode *root) {

  std::vector<OctTreeNode *> palette;

  std::queue<OctTreeNode *> q;
  q.push(root);

  while (!q.empty()) {

    OctTreeNode *curr = q.front();
    q.pop();

    if (curr == nullptr)
      continue;

    if (curr->isLeaf())
      palette.push_back(curr);
    else {

      for (OctTreeNode *node : curr->getChildren()) {
        q.push(node);
      }
    }
  }

  return palette;
}


static void reduce(OctTreeNode *root, int palette_size, int leaves) {
  std::stack<OctTreeNode *> stack;

  stack.push(root);

  while (!stack.empty()) {

      if(leaves == palette_size) return;
    OctTreeNode *curr = stack.top();
    stack.pop();
    if (curr == nullptr)
      continue;

    for (OctTreeNode *node : curr->getChildren()) {
      if(!node) continue;
      node->reduce();
      leaves--;
    }
  }
 }

std::vector<Color *> oct_tree_gen(std::vector<Color *> colors, int size) {

  OctTreeNode *root = new OctTreeNode(0);

  for (int i = 0; i < colors.size(); i++) {
	root->insert(colors[i]);
    
  }

  std::vector<Color *> palette;
  std::vector<OctTreeNode *> leaves = get_leaf_nodes(root);
  reduce(root, size, leaves.size());
  std::vector<OctTreeNode * > final_leaves = get_leaf_nodes(root);
  std::cout << final_leaves.size() << std::endl; 

  for (const OctTreeNode *node : final_leaves) {
     palette.push_back(node->color);
  }
  return palette;
}
