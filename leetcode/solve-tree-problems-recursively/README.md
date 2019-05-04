# Solve Problems Recursively

In previous sections, we have introduced how to solve tree traversal problem recursively. Recursion is one of the most powerful and frequent used methods for solving tree related problems.

As we know, a tree can be defined recursively as a node\(the root node\), which includes a value and a list of references to other nodes. Recursion is one of the nature features of a tree. Therefore, many tree problems can be solved recursively. For each recursion level, we can only focus on the problem within one single node and call the function recursively to solve its children.

Typically, we can solve a tree problem recursively from the top down or from the bottom up.

#### "Top-down" Solution <a id="top-down-solution"></a>

"Top-down" means that in each recursion level, we will visit the node first to come up with some values, and pass these values to its children when calling the function recursively. So the "top-down" solution can be considered as kind of **preorder** traversal. To be specific, the recursion function `top_down(root, params)`works like this:

```text
1. return specific value for null node
2. update the answer if needed                      // answer <-- params
3. left_ans = top_down(root.left, left_params)      // left_params <-- root.val, params
4. right_ans = top_down(root.right, right_params)   // right_params <-- root.val, params 
5. return the answer if needed                      // answer <-- left_ans, right_ans
```

For instance, consider this problem: given a binary tree, find its maximum depth.

We know that the depth of the root node is `1`. For each node, if we know the depth of the node, we will know the depth of its children. Therefore, if we pass the depth of the node as a parameter when calling the function recursively, all the nodes know the depth of themselves. And for leaf nodes, we can use the depth to update the final answer. Here is the pseudocode for the recursion function `maximum_depth(root, depth)`:

```text
1. return if root is null
2. if root is a leaf node:
3.      answer = max(answer, depth)         // update the answer if needed
4. maximum_depth(root.left, depth + 1)      // call the function recursively for left child
5. maximum_depth(root.right, depth + 1)     // call the function recursively for right child
```

```cpp
int answer;		       // don't forget to initialize answer before call maximum_depth
void maximum_depth(TreeNode* root, int depth) {
    if (!root) {
        return;
    }
    if (!root->left && !root->right) {
        answer = max(answer, depth);
    }
    maximum_depth(root->left, depth + 1);
    maximum_depth(root->right, depth + 1);
}
```

#### "Bottom-up" Solution <a id="bottom-up-solution"></a>

"Bottom-up" is another recursion solution. In each recursion level, we will firstly call the functions recursively for all the children nodes and then come up with the answer according to the return values and the value of the root node itself. This process can be regarded as kind of **postorder** traversal. Typically, a "bottom-up" recursion function `bottom_up(root)` will be like this:

```text
1. return specific value for null node
2. left_ans = bottom_up(root.left)          // call function recursively for left child
3. right_ans = bottom_up(root.right)        // call function recursively for right child
4. return answers                           // answer <-- left_ans, right_ans, root.val
```

Let's go on discussing the question about maximum depth but using a different way of thinking: for a single node of the tree, what will be the maximum depth `x` of the subtree rooted at itself?

If we know the maximum depth `l` of the subtree rooted at its **left** child and the maximum depth `r` of the subtree rooted at its **right** child, can we answer the previous question? Of course yes, we can choose the maximum between them and plus 1 to get the maximum depth of the subtree rooted at the selected node. That is `x = max(l, r) + 1`.

It means that for each node, we can get the answer after solving the problem of its children. Therefore, we can solve this problem using a "bottom-up" solution. Here is the pseudocode for the recursion function `maximum_depth(root)`:

```text
1. return 0 if root is null                 // return 0 for null node
2. left_depth = maximum_depth(root.left)
3. right_depth = maximum_depth(root.right)
4. return max(left_depth, right_depth) + 1  // return depth of the subtree rooted at root
```

```cpp
int maximum_depth(TreeNode* root) {
	if (!root) {
		return 0;                                 // return 0 for null node
	}
	int left_depth = maximum_depth(root->left);	
	int right_depth = maximum_depth(root->right);
	return max(left_depth, right_depth) + 1;	  // return depth of the subtree rooted at root
}
```

#### Conclusion <a id="conclusion"></a>

It is not easy to understand recursion and find out a recursion solution for the problem.

When you meet a tree problem, ask yourself two questions: can you determine some parameters to help the node know the answer of itself? Can you use these parameters and the value of the node itself to determine what should be the parameters parsing to its children? If the answers are both yes, try to solve this problem using a "`top-down`" recursion solution.

Or you can think the problem in this way: for a node in a tree, if you know the answer of its children, can you calculate the answer of the node? If the answer is yes, solving the problem recursively from `bottom up` might be a good way.

In the following sections, we provide several classic problems for you to help you understand tree structure and recursion better.

