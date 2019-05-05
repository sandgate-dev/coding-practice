# Binary Tree Level Order Traversal

Level-order traversal is to traverse the tree level by level.

`Breadth-First Search` is an algorithm to traverse or search in data structures like a tree or a graph. The algorithm starts with a root node and visit the node itself first. Then traverse its neighbors, traverse its second level neighbors, traverse its third level neighbors, so on and so forth.

When we do breadth-first search in a tree, the order of the nodes we visited is in level order. Typically, we use a queue to help us to do BFS. If you are not so familiar with the queue.

## Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. \(ie, from left to right, level by level\).

For example:  
Given binary tree `[3,9,20,null,null,15,7]`,

```text
    3
   / \
  9  20
    /  \
   15   7
```

return its level order traversal as:

```text
[
  [3],
  [9,20],
  [15,7]
]
```

```python
def levelOrder(self, root):
    result = []
    
    if not root:
        return result
    
    curr_level = [root]
    while curr_level:
        level_result = []
        next_level = []
        
        for curr in curr_level:
            level_result.append(curr.val)
            if curr.left:
                next_level.append(curr.left)
            if curr.right:
                next_level.append(curr.right)
                
        result.append(level_result)
        curr_level = next_level
        
    return result

```

