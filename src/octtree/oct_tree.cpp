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

bool should_reduce(OctTreeNode *node, int palette_size, int leaves) {

  if(node==nullptr) return false;

  int count = 0;


  // if the node has children that are not leaves, we shouldn't reduce
for (OctTreeNode *child : node->getChildren()) {
    if (child != nullptr && child->isLeaf())
        count++;
}

if (count == 0){
	return false;
}
  // if the colors of all the children vary too much, we also shouldn't reduce

  /* *
   * We'll calculate the variance of the colors, and compare it to a threshold.
   *
   * since the children of this node are a small set of colors in the image,
   * we have to use the sample variance.
   * Also, we have a 3D space (RGB), so we need to use multivariate variance.
   *
   *
   * Since each node has 8 children, we will always be dealing
   * with at most a constant 8 items (they might be null, in that case we
   * discard them). That means this operation will take O(1) time.
   * */

  double cov[3][3] = {0};
  int ave_r = 0;
  int ave_g = 0;
  int ave_b = 0;


  for (OctTreeNode *child : node->getChildren()) {
    if (child == nullptr || !child->isLeaf() )
      continue;
    Color *col = child->color;
    ave_r += col->Red();
    ave_g += col->Green();
    ave_b += col->Blue();
  }

  ave_r /= count;
  ave_g /= count;
  ave_b /= count;

  for (OctTreeNode *child : node->getChildren()) {
    if(child==nullptr) continue;
    Color *col = child->color;
    cov[0][0] += pow(col->Red() - ave_r, 2);
    cov[1][1] += pow(col->Green() - ave_g, 2);
    cov[2][2] += pow(col->Blue() - ave_b, 2);
    cov[0][1] += (col->Red() - ave_r) * (col->Green() - ave_g);
    cov[0][2] += (col->Red() - ave_r) * (col->Blue() - ave_b);
    cov[1][2] += (col->Green() - ave_g) * (col->Blue() - ave_b);
  }



cov[1][0] = cov[0][1];
cov[2][0] = cov[0][2];
cov[2][1] = cov[1][2];

for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
        cov[i][j] /= count;



double det = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1]) -
             cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]) +
             cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);


return det >= THRESHOLD;
}

static void reduce(OctTreeNode *root, int palette_size, int leaves) {

  std::stack<OctTreeNode *> stack;

  stack.push(root);

  while (!stack.empty() && leaves > palette_size) {

    OctTreeNode *curr = stack.top();
    stack.pop();
    if (curr == nullptr)
      continue;

    for (OctTreeNode *node : curr->getChildren()) {
      if (!should_reduce(node, palette_size, leaves)){
       	      stack.push(node);
	      continue;
      }
      node->reduce();
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
  for (const OctTreeNode *node : leaves) {
    // palette.push_back(node->color);
  }
  return palette;
}
