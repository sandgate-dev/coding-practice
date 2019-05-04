# Path Sum

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

**Note:** A leaf is a node with no children.

**Example:**

Given the below binary tree and `sum = 22`,

```text
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1
```

return true, as there exist a root-to-leaf path `5->4->11->2` which sum is 22.

```python
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    if root == None:
        return False
    else:
        current = root
        result = []
        result.append(current)
        result.append(current.val)

        while result != []:
            pathsum = result.pop()
            current = result.pop()

            if not current.left and not current.right:
                if pathsum == sum:
                    return True

            if current.right:
                r_pathsum = pathsum + current.right.val
                result.append(current.right)
                result.append(r_pathsum)

            if current.left:
                l_pathsum = pathsum + current.left.val
                result.append(current.left)
                result.append(l_pathsum)

        return False

```

