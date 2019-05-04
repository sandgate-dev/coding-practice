# Symmetric Tree

Given a binary tree, check whether it is a mirror of itself \(ie, symmetric around its center\).

For example, this binary tree `[1,2,2,3,4,4,3]` is symmetric:

```text
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

But the following `[1,2,2,null,3,null,3]` is not:

```text
    1
   / \
  2   2
   \   \
   3    3
```

**Note:**  
Bonus points if you could solve it both recursively and iteratively.

```python
def isSymmetric(self, root: TreeNode) -> bool:
    def isMirror(left, right):
        if left is None or right is None:
            return left == None and right == None

        return (
            left.val == right.val
            and isMirror(left.left, right.right)
            and isMirror(left.right, right.left)
        )

    return isMirror(root, root)
```

