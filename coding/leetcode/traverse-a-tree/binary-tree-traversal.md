# Binary Tree Traversal

## Preorder

Given a binary tree, return the _preorder_ traversal of its nodes' values.

**Example:**

```text
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1, 2, 3]
```

**Follow up:** Recursive solution is trivial, could you do it iteratively?

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def preorderTraversal(self, root: TreeNode) -> List[int]:
    if root == None:
        return []
    else:
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

```

## Inorder

```text
Output: [1, 3, 2]
```

```python
def inorderTraversal(self, root: TreeNode) -> List[int]:
    if root:
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
    else:
        return []  
```

## Postorder

```text
Output: [3, 2, 1]
```

```python
def postorderTraversal(self, root: TreeNode) -> List[int]:
    if root:
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
    else:
        return []
```

