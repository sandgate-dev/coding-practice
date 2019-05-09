---
description: By Dan Bader
---

# Linked Lists in Python

Learn how to implement a linked list data structure in Python, using only built-in data types and functionality from the standard library.

Every Python programmer should know about linked lists:

They are among the simplest and most common data structures used in programming.

So, if you ever found yourself wondering, **“Does Python have a built-in or ‘native’ linked list data structure?”** or, **“How do I write a linked list in Python?”** then this tutorial will help you.

Python doesn’t ship with a built-in linked list data type in the “classical” sense. Python’s `list` type is implemented as a dynamic array—which means it doesn’t suit the typical scenarios where you’d want to use a “proper” linked list data structure for performance reasons.

Please note that this tutorial only considers linked list implementations that work on a “plain vanilla” Python install. I’m leaving out third-party packages intentionally. They don’t apply during coding interviews and it’s difficult to keep an up-to-date list that considers all packages available on Python packaging repositories.

Before we get into the weeds and look at linked list implementations in Python, let’s do a quick recap of what a linked list data structure is—and how it compares to an array.

### What are the characteristics of a linked list?

A linked list is an ordered collection of values. Linked lists are similar to arrays in the sense that they contain objects in a linear order. However they [differ from arrays in their memory layout](https://dbader.org/blog/python-arrays).

**Arrays** are _contiguous_ data structures and they’re composed of fixed-size data records stored in adjoining blocks of memory. [In an array, data is tightly packed](https://dbader.org/blog/python-arrays)—and we know the size of each data record which allows us to quickly look up an element given its index in the array:

![Array Visualization](https://dbader.org/static/figures/python-linked-list-array-visualization.jpeg)

**Linked lists**, however, are made up of data records _linked_ together by pointers. This means that the data records that hold the actual “data payload” can be stored anywhere in memory—what creates the linear ordering is how each data record “points” to the next one:

![Linked List Visualization](https://dbader.org/static/figures/python-linked-list-visualization.jpeg)

There are two different kinds of linked lists: **singly-linked lists** and **doubly-linked lists**. What you saw in the previous example was a singly-linked list—each element in it has a reference to \(a “pointer”\) to the _next_ element in the list.

In a doubly-linked list, each element has a reference to both the _next and the previous_ element. Why is this useful? Having a reference to the previous element can speed up some operations, like removing \(“unlinking”\) an element from a list or traversing the list in reverse order.

### How do linked lists and arrays compare performance-wise?

You just saw how linked lists and arrays use different data layouts behind the scenes to store information. This data layout difference reflects in the performance characteristics of linked lists and arrays:

* **Element Insertion & Removal**: Inserting and removing elements from a \(doubly\) linked list has time complexity _O\(1\)_, whereas doing the same on an array requires an _O\(n\)_ copy operation in the worst case. On a linked list we can simply “hook in” a new element anywhere we want by adjusting the pointers from one data record to the next. On an array we have to allocate a bigger storage area first and copy around the existing elements, leaving a blank space to insert the new element into.
* **Element Lookup**: Similarly, looking up an element given its index is a slow _O\(n\)_ time operation on a linked list—but a fast _O\(1\)_ lookup on an array. With a linked list we must jump from element to element and search the structure from the “head” of the list to find the index we want. But with an array we can calculate the exact address of an element in memory based on its index and the \(fixed\) size of each data record.
* **Memory Efficiency**: Because the data stored in arrays is tightly packed they’re generally more space-efficient than linked lists. This mostly applies to static arrays, however. [Dynamic arrays](https://en.wikipedia.org/wiki/Dynamic_array) typically over-allocate their backing store slightly to speed up element insertions in the average case, thus increasing the memory footprint.

Now, how does this performance difference come into play with Python? Remember that [Python’s built-in `list` type is in fact a dynamic array](http://www.laurentluce.com/posts/python-list-implementation/). This means the performance differences we just discussed apply to it. Likewise, Python’s immutable `tuple` data type can be considered a static array in this case—with similar performance trade-offs compared to a proper linked list.

### Does Python have a built-in or “native” linked list data structure?

Let’s come back to the original question. If you want to use a linked list in Python, is there a built-in data type you can use directly?

The answer is: “It depends.”

As of Python 3.6 \(CPython\), doesn’t provide a dedicated linked list data type. There’s nothing like [Java’s `LinkedList`](https://docs.oracle.com/javase/7/docs/api/java/util/LinkedList.html) built into Python or into the Python standard library.

Python does however include the `collections.deque` class which provides a [double-ended queue](https://dbader.org/blog/queues-in-python) and is implemented as a doubly-linked list internally. Under some specific circumstances you might be able to use it as a “makeshift” linked list. If that’s not an option you’ll need to write your own linked list implementation from scratch.

### How do I write a linked list using Python?

If you want to stick with functionality built into the core language and into the Python standard library you have two options for implementing a linked list:

* You could either use the `collections.deque` class from the Python standard library and take advantage of the fact that it’s implemented as a doubly-linked list behind the scenes. But this will only work for some use cases—I’ll go into more details on that further down in the article.
* Alternatively, you could define your own linked list type in Python by writing it from scratch using other built-in data types. You’d implement your own custom linked list class or base your implementation of Lisp-style chains of `tuple` objects. Again, see below for more details.

Now that we’ve covered some general questions on linked lists and their availability in Python, read on for examples of how to make both of the above approaches work.

### Option 1: Using `collections.deque` as a Linked List

This approach might seem a little odd at first because the `collections.deque`class implements a double-ended queue, and it’s typically used as the go-to [stack](https://dbader.org/blog/stacks-in-python) or [queue implementation in Python](https://dbader.org/blog/queues-in-python).

But using this class as a “makeshift” linked list might make sense under some circumstances. You see, CPython’s `deque` is [powered by a doubly-linked list](https://github.com/python/cpython/blob/947629916a5ecb1f6f6792e9b9234e084c5bf274/Modules/_collectionsmodule.c#L24-L26) behind the scenes and it provides a full “list-like” set of functionality.

Under _some_ circumstances, this makes treating `deque` objects as linked list replacements a valid option. Here are some of the key performance characteristics of this approach:

* **Inserting and removing elements at the front and back of a `deque` is a fast** _**O\(1\)**_ **operation.** However, inserting or removing in the middle takes _O\(n\)_time because we don’t have access to the previous-element or next-element linked list pointers. That’s abstracted away by the `deque` interface.
* **Storage is** _**O\(n\)**_**—but not every element gets its own list node.** The `deque`class uses blocks that hold multiple elements at once and then these blocks are linked together as a doubly-linked list. [As of CPython 3.6 the block size is 64 elements.](https://github.com/python/cpython/blob/947629916a5ecb1f6f6792e9b9234e084c5bf274/Modules/_collectionsmodule.c#L14-L22) This incurs some space overhead but retains the general performance characteristics given a large enough number of elements.
* **In-place reversal**: In Python 3.2+ the elements in a `deque` instance can be reversed in-place with the `reverse()` method. This takes _O\(n\)_ time and no extra space.

Using `collections.deque` as a linked list in Python can be a valid choice if you mostly care about insertion performance at the beginning or the end of the list, and you don’t need access to the previous-element and next-element pointers on each object directly.

Don’t use a `deque` if you need _O\(1\)_ performance when removing elements. Removing elements by key or by index requires an _O\(n\)_ search, even if you have already have a reference to the element to be removed. This is the main downside of using a `deque` like a linked list.

If you’re looking for a linked list in Python because you want to implement [queues](https://dbader.org/blog/queues-in-python) or a [stacks](https://dbader.org/blog/stacks-in-python) then a `deque` is a great choice, however.

Here are some examples on how you can use Python’s `deque` class as a replacement for a linked list:

```python
>>> import collections
>>> lst = collections.deque()

# Inserting elements at the front
# or back takes O(1) time:
>>> lst.append('B')
>>> lst.append('C')
>>> lst.appendleft('A')
>>> lst
deque(['A', 'B', 'C'])

# However, inserting elements at
# arbitrary indexes takes O(n) time:
>>> lst.insert(2, 'X')
>>> lst
deque(['A', 'B', 'X', 'C'])

# Removing elements at the front
# or back takes O(1) time:
>>> lst.pop()
'C'
>>> lst.popleft()
'A'
>>> lst
deque(['B', 'X'])

# Removing elements at arbitrary
# indexes or by key takes O(n) time again:
>>> del lst[1]
>>> lst.remove('B')

# Deques can be reversed in-place:
>>> lst = collections.deque(['A', 'B', 'X', 'C'])
>>> lst.reverse()
deque(['C', 'X', 'B', 'A'])

# Searching for elements takes
# O(n) time:
>>> lst.index('X')
1
```

### Option 2: Writing Your Own Python Linked Lists

If you need full control over the layout of each linked list node then there’s no perfect solution available in the Python standard library. If you want to stick with the standard library and built-in data types then writing your own linked list is your best bet.

You’ll have to make a choice between implementing a singly-linked or a doubly-linked list. I’ll give examples of both, including some of the common operations like how to search for elements, or how to reverse a linked list.

Let’s take a look at two concrete Python linked list examples. One for a singly-linked list, and one for a double-linked list.

### ✅ A Singly-Linked List Class in Python

Here’s how you might implement **a class-based singly-linked list in Python**, including some of the standard algorithms:

```python
class ListNode:
    """
    A node in a singly-linked list.
    """
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return repr(self.data)


class SinglyLinkedList:
    def __init__(self):
        """
        Create a new singly-linked list.
        Takes O(1) time.
        """
        self.head = None

    def __repr__(self):
        """
        Return a string representation of the list.
        Takes O(n) time.
        """
        nodes = []
        curr = self.head
        while curr:
            nodes.append(repr(curr))
            curr = curr.next
        return '[' + ', '.join(nodes) + ']'

    def prepend(self, data):
        """
        Insert a new element at the beginning of the list.
        Takes O(1) time.
        """
        self.head = ListNode(data=data, next=self.head)

    def append(self, data):
        """
        Insert a new element at the end of the list.
        Takes O(n) time.
        """
        if not self.head:
            self.head = ListNode(data=data)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(data=data)

    def find(self, key):
        """
        Search for the first element with `data` matching
        `key`. Return the element or `None` if not found.
        Takes O(n) time.
        """
        curr = self.head
        while curr and curr.data != key:
            curr = curr.next
        return curr  # Will be None if not found

    def remove(self, key):
        """
        Remove the first occurrence of `key` in the list.
        Takes O(n) time.
        """
        # Find the element and keep a
        # reference to the element preceding it
        curr = self.head
        prev = None
        while curr and curr.data != key:
            prev = curr
            curr = curr.next
        # Unlink it from the list
        if prev is None:
            self.head = curr.next
        elif curr:
            prev.next = curr.next
            curr.next = None

    def reverse(self):
        """
        Reverse the list in-place.
        Takes O(n) time.
        """
        curr = self.head
        prev_node = None
        next_node = None
        while curr:
            next_node = curr.next
            curr.next = prev_node
            prev_node = curr
            curr = next_node
        self.head = prev_node
```

And here’s how you’d use this linked list class in practice:

```python
>>> lst = SinglyLinkedList()
>>> lst
[]

>>> lst.prepend(23)
>>> lst.prepend('a')
>>> lst.prepend(42)
>>> lst.prepend('X')
>>> lst.append('the')
>>> lst.append('end')

>>> lst
['X', 42, 'a', 23, 'the', 'end']

>>> lst.find('X')
'X'
>>> lst.find('y')
None

>>> lst.reverse()
>>> lst
['end', 'the', 23, 'a', 42, 'X']

>>> lst.remove(42)
>>> lst
['end', 'the', 23, 'a', 'X']

>>> lst.remove('not found')
```

Note that removing an element in this implementation is still an _O\(n\)_ time operation, even if you already have a reference to a `ListNode` object.

In a singly-linked list removing an element typically requires searching the list because we need to know the previous and the next element. With a double-linked list you could write a `remove_elem()` method that unlinks and removes a node from the list in _O\(1\)_ time.

### ✅ A Doubly-Linked List Class in Python

Let’s have a look at **how to implement a doubly-linked list in Python**. The following `DoublyLinkedList` class should point you in the right direction:

```python
class DListNode:
    """
    A node in a doubly-linked list.
    """
    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    def __repr__(self):
        return repr(self.data)


class DoublyLinkedList:
    def __init__(self):
        """
        Create a new doubly linked list.
        Takes O(1) time.
        """
        self.head = None

    def __repr__(self):
        """
        Return a string representation of the list.
        Takes O(n) time.
        """
        nodes = []
        curr = self.head
        while curr:
            nodes.append(repr(curr))
            curr = curr.next
        return '[' + ', '.join(nodes) + ']'

    def prepend(self, data):
        """
        Insert a new element at the beginning of the list.
        Takes O(1) time.
        """
        new_head = DListNode(data=data, next=self.head)
        if self.head:
            self.head.prev = new_head
        self.head = new_head

    def append(self, data):
        """
        Insert a new element at the end of the list.
        Takes O(n) time.
        """
        if not self.head:
            self.head = DListNode(data=data)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = DListNode(data=data, prev=curr)

    def find(self, key):
        """
        Search for the first element with `data` matching
        `key`. Return the element or `None` if not found.
        Takes O(n) time.
        """
        curr = self.head
        while curr and curr.data != key:
            curr = curr.next
        return curr  # Will be None if not found

    def remove_elem(self, node):
        """
        Unlink an element from the list.
        Takes O(1) time.
        """
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self.head:
            self.head = node.next
        node.prev = None
        node.next = None

    def remove(self, key):
        """
        Remove the first occurrence of `key` in the list.
        Takes O(n) time.
        """
        elem = self.find(key)
        if not elem:
            return
        self.remove_elem(elem)

    def reverse(self):
        """
        Reverse the list in-place.
        Takes O(n) time.
        """
        curr = self.head
        prev_node = None
        while curr:
            prev_node = curr.prev
            curr.prev = curr.next
            curr.next = prev_node
            curr = curr.prev
        self.head = prev_node.prev
```

Here are a few examples on how to use this class. Notice how we can now remove elements in _O\(1\)_ time with the `remove_elem()` function if we already hold a reference to the list node representing the element:

```python
>>> lst = DoublyLinkedList()
>>> lst
[]

>>> lst.prepend(23)
>>> lst.prepend('a')
>>> lst.prepend(42)
>>> lst.prepend('X')
>>> lst.append('the')
>>> lst.append('end')

>>> lst
['X', 42, 'a', 23, 'the', 'end']

>>> lst.find('X')
'X'
>>> lst.find('y')
None

>>> lst.reverse()
>>> lst
['end', 'the', 23, 'a', 42, 'X']

>>> elem = lst.find(42)
>>> lst.remove_elem(elem)

>>> lst.remove('X')
>>> lst.remove('not found')
>>> lst
['end', 'the', 23, 'a']
```

Both example for Python linked lists you saw here were class-based. An alternative approach would be to implement a _Lisp-style_ linked list in Python using tuples as the core building blocks \(“cons pairs”\). Here’s a tutorial that goes into more detail: [Functional Linked Lists in Python](https://dbader.org/blog/functional-linked-lists-in-python).

### Python Linked Lists: Recap & Recommendations

We just looked at a number of approaches to implement a singly- and doubly-linked list in Python. You also saw some code examples of the standard operations and algorithms, for example how to reverse a linked list in-place.

You should only consider using a linked list in Python when you’ve determined that you absolutely need a linked data structure for performance reasons \(or you’ve been asked to use one in a coding interview.\)

In many cases the same algorithm implemented on top of Python’s highly optimized `list` objects will be sufficiently fast. If you know a dynamic array won’t cut it and you need a linked list, then check first if you can take advantage of Python’s built-in `deque` class.

If none of these options work for you, and you want to stay within the standard library, only then should you write your own Python linked list.

In an interview situation I’d also advise you to write your own implementation from scratch because that’s usually what the interviewer wants to see. However it can be beneficial to mention that `collections.deque` offers similar performance under the right circumstances. Good luck and…Happy Pythoning!

[_Read the full “Fundamental Data Structures in Python” article series here_](https://dbader.org/blog/fundamental-data-structures-in-python)_. This article is missing something or you found an error? Help a brother out and leave a comment below._

## Functional linked lists in Python

By Dan Bader

Linked lists are fundamental data structures that every programmer should know. This article explains how to implement a simple linked list data type in Python using a functional programming style.

### Inspiration

The excellent book [Programming in Scala](http://amzn.to/Rox4kI) inspired me to play with functional programming concepts in Python. I ended up implementing a basic [linked list](https://en.wikipedia.org/wiki/Linked_list) __data structure using a Lisp-like functional style that I want to share with you.

I wrote most of this using [Pythonista](http://omz-software.com/pythonista/) on my iPad. Pythonista is a Python IDE-slash-scratchpad and surprisingly fun to work with. It’s great when you’re stuck without a laptop and want to explore some CS fundamentals :\)

So without further ado, let’s dig into the implementation.

### Constructing linked lists

Our linked list data structure consists of two fundamental building blocks: `Nil`and `cons`. `Nil` represents the empty list and serves as a sentinel for longer lists. The `cons` operation extends a list at the front by inserting a new value.

The lists we construct using this method consist of nested 2-tuples. For example, the list `[1, 2, 3]` is represented by the expression `cons(1, cons(2, cons(3, Nil)))` which evaluates to the nested tuples `(1, (2, (3, Nil)))`.

```text
Nil = None

def cons(x, xs=Nil):
    return (x, xs)

assert cons(0) == (0, Nil)
assert cons(0, (1, (2, Nil))) == (0, (1, (2, Nil)))
```

Why should we use this structure?

First, the [cons operation](http://en.wikipedia.org/wiki/Cons) is deeply rooted in the history of functional programming. From Lisp’s _cons cells_ to ML’s and Scala’s `::` operator, cons is everywhere – you can even use it as a verb.

Second, tuples are a convenient way to define simple data structures. For something as simple as our list building blocks, we don’t necessarily have to define a proper class. Also, it keeps this introduction short and sweet.

Third, tuples are _immutable_ in Python which means their state cannot be modified after creation. [Immutability](http://en.wikipedia.org/wiki/Immutable_object) is often a desired property because it helps you write simpler and more thread-safe code. I like [this article](http://www.altdevblogaday.com/2012/04/26/functional-programming-in-c/) by John Carmack where he shares his views on functional programming and immutability.

Abstracting away the tuple construction using the `cons` function gives us a lot of flexibility on how lists are represented internally as Python objects. For example, instead of using 2-tuples we could store our elements in a chain of anonymous functions with Python’s `lambda` keyword.

```text
def cons(x, xs=Nil):
    return lambda i: x if i == 0 else xs
```

To write simpler tests for more complex list operations we’ll introduce the helper function `lst`. It allows us to define list instances using a more convenient syntax and without deeply nested `cons` calls.

```text
def lst(*xs):
    if not xs:
        return Nil
    else:
        return cons(xs[0], lst(*xs[1:]))

assert lst() == Nil
assert lst(1) == (1, Nil)
assert lst(1, 2, 3, 4) == (1, (2, (3, (4, Nil))))
```

### Basic operations

All operations on linked lists can be expressed in terms of the three fundamental operations `head`, `tail`, and `is_empty`.

* `head` returns the first element of a list.
* `tail` returns a list containing all elements except the first.
* `is_empty` returns `True` if the list contains zero elements.

You’ll see later that these three operations are enough to implement a simple sorting algorithm like _insertion sort_.

```text
def head(xs):
    return xs[0]

assert head(lst(1, 2, 3)) == 1
```

```text
def tail(xs):
    return xs[1]

assert tail(lst(1, 2, 3, 4)) == lst(2, 3, 4)
```

```text
def is_empty(xs):
    return xs is Nil

assert is_empty(Nil)
assert not is_empty(lst(1, 2, 3))
```

### Length and concatenation

The `length` operation returns the number of elements in a given list. To find the length of a list we need to scan all of its _n_ elements. Therefore this operation has a time complexity of _O\(n\)_.

```text
def length(xs):
    if is_empty(xs):
        return 0
    else:
        return 1 + length(tail(xs))

assert length(lst(1, 2, 3, 4)) == 4
assert length(Nil) == 0
```

`concat` takes two lists as arguments and concatenates them. The result of `concat(xs, ys)` is a new list that contains all elements in `xs` followed by all elements in `ys`. We implement the function with a simple divide and conquer algorithm.

```text
def concat(xs, ys):
    if is_empty(xs):
        return ys
    else:
        return cons(head(xs), concat(tail(xs), ys))

assert concat(lst(1, 2), lst(3, 4)) == lst(1, 2, 3, 4)
```

### Last, init, and list reversal

The basic operations `head` and `tail` have corresponding operations `last` and `init`. `last` returns the last element of a non-empty list and `init` returns all elements except the last one \(the _initial_ elements\).

```text
def last(xs):
    if is_empty(tail(xs)):
        return head(xs)
    else:
        return last(tail(xs))

assert last(lst(1, 3, 3, 4)) == 4
```

```text
def init(xs):
    if is_empty(tail(tail(xs))):
        return cons(head(xs))
    else:
        return cons(head(xs), init(tail(xs)))

assert init(lst(1, 2, 3, 4)) == lst(1, 2, 3)
```

Both operations need _O\(n\)_ time to compute their result. Therefore it’s a good idea to reverse a list if you frequently use `last` or `init` to access its elements. The `reverse` function below implements list reversal, but in a slow way that takes _O\(n²\)_ time.

```text
def reverse(xs):
    if is_empty(xs):
        return xs
    else:
        return concat(reverse(tail(xs)), cons(head(xs), Nil))

assert reverse(Nil) == Nil
assert reverse(cons(0, Nil)) == (0, Nil)
assert reverse(lst(1, 2, 3, 4)) == lst(4, 3, 2, 1)
assert reverse(reverse(lst(1, 2, 3, 4))) == lst(1, 2, 3, 4)
```

### Prefixes and suffixes

The following operations `take` and `drop` generalize `head` and `tail` by returning arbitrary prefixes and suffixes of a list. For example, `take(2, xs)` returns the first two elements of the list `xs` whereas `drop(3, xs)` returns everything except the last three elements in `xs`.

```text
def take(n, xs):
    if n == 0:
        return Nil
    else:
        return cons(head(xs), take(n-1, tail(xs)))

assert take(2, lst(1, 2, 3, 4)) == lst(1, 2)
```

```text
def drop(n, xs):
    if n == 0:
        return xs
    else:
        return drop(n-1, tail(xs))

assert drop(1, lst(1, 2, 3)) == lst(2, 3)
assert drop(2, lst(1, 2, 3, 4)) == lst(3, 4)
```

### Element selection

Random element selection on linked lists doesn’t really make sense in terms of time complexity – accessing an element at index _n_ requires _O\(n\)_ time. However, the element access operation `apply` is simple to implement using `head` and `drop`.

```text
def apply(i, xs):
    return head(drop(i, xs))

assert apply(0, lst(1, 2, 3, 4)) == 1
assert apply(2, lst(1, 2, 3, 4)) == 3
```

### More complex examples

The three basic operations `head`, `tail`, and `is_empty` are all we need to implement a simple \(and slow\) sorting algorithm like [insertion sort](http://en.wikipedia.org/wiki/Insertion_sort).

```text
def insert(x, xs):
    if is_empty(xs) or x <= head(xs):
        return cons(x, xs)
    else:
        return cons(head(xs), insert(x, tail(xs)))

assert insert(0, lst(1, 2, 3, 4)) == lst(0, 1, 2, 3, 4)
assert insert(99, lst(1, 2, 3, 4)) == lst(1, 2, 3, 4, 99)
assert insert(3, lst(1, 2, 4)) == lst(1, 2, 3, 4)

def isort(xs):
    if is_empty(xs):
        return xs
    else:
        return insert(head(xs), isort(tail(xs)))

assert isort(lst(1, 2, 3, 4)) == lst(1, 2, 3, 4)
assert isort(lst(3, 1, 2, 4)) == lst(1, 2, 3, 4)
```

The following `to_string` operation flattens the recursive structure of a given list and returns a Python-style string representation of its elements. This is useful for debugging and makes for a nice little programming exercise.

```text
def to_string(xs, prefix="[", sep=", ", postfix="]"):
    def _to_string(xs):
        if is_empty(xs):
            return ""
        elif is_empty(tail(xs)):
            return str(head(xs))
        else:
            return str(head(xs)) + sep + _to_string(tail(xs))
    return prefix + _to_string(xs) + postfix

assert to_string(lst(1, 2, 3, 4)) == "[1, 2, 3, 4]"
```

### Where to go from here

This article is more of a thought experiment than a guide on how to implement a useful linked list in Python. Keep in mind that the above code has severe restrictions and is not fit for real life use. For example, if you use this linked list implementation with larger example lists you’ll quickly hit recursion depth limits \(CPython doesn’t optimize tail recursion\).

I spent a few fun hours playing with functional programming concepts in Python and I hope I inspired you to do the same. If you want to explore functional programming in ‘real world’ Python check out the following resources:

* [The Python Functional Programming HOWTO](http://docs.python.org/3.3/howto/functional.html)
* [Charming Python: Functional programming in Python](http://www.ibm.com/developerworks/linux/library/l-prog/index.html)
* [Mike Müller’s PyCon talk: Functional programming with Python](http://pyvideo.org/video/1799/functional-programming-with-python)

