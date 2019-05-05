# Queue & Stack

In this chapter, we will first introduce `First-in-first-out` \(FIFO\) and how it works in a `queue`.

The goal of this chapter is to help you:

1. Understand the `definition` of FIFO and queue;
2. Be able to `implement` a queue by yourself;
3. Be familiar with the `built-in queue structure`;
4. Use queue to solve simple problems.

## First-in-first-out Data Structure

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/05/03/screen-shot-2018-05-03-at-151021.png)

In a FIFO data structure, `the first element added to the queue will be processed first`.

As shown in the picture above, the queue is a typical FIFO data structure. The insert operation is also called enqueue and the new element is always `added at the end of the queue`. The delete operation is called dequeue. You are only allowed to `remove the first element.`

## Queue - Implementation

To implement a queue, we may use a dynamic array and an index pointing to the head of the queue.

As mentioned, a queue should support two operations: `enqueue` and `dequeue`. Enqueue appends a new element to the queue while dequeue removes the first element. So we need an index to indicate the starting point.

Here is an implementation for your reference:

```java
// "static void main" must be defined in a public class.

class MyQueue {
    // store elements
    private List<Integer> data;         
    // a pointer to indicate the start position
    private int p_start;            
    public MyQueue() {
        data = new ArrayList<Integer>();
        p_start = 0;
    }
    /** Insert an element into the queue. Return true if the operation is successful. */
    public boolean enQueue(int x) {
        data.add(x);
        return true;
    };    
    /** Delete an element from the queue. Return true if the operation is successful. */
    public boolean deQueue() {
        if (isEmpty() == true) {
            return false;
        }
        p_start++;
        return true;
    }
    /** Get the front item from the queue. */
    public int Front() {
        return data.get(p_start);
    }
    /** Checks whether the queue is empty or not. */
    public boolean isEmpty() {
        return p_start >= data.size();
    }     
};

public class Main {
    public static void main(String[] args) {
        MyQueue q = new MyQueue();
        q.enQueue(5);
        q.enQueue(3);
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
        q.deQueue();
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
        q.deQueue();
        if (q.isEmpty() == false) {
            System.out.println(q.Front());
        }
    }
}
```

#### Drawback

The implementation above is straightforward but is inefficient in some cases. With the movement of the start pointer, more and more space is wasted. And it will be unacceptable when we only have a space limitation.

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/21/screen-shot-2018-07-21-at-153558.png)

Let's consider a situation when we are only able to allocate an array whose maximum length is 5. Our solution works well when we have only added less than 5 elements. For example, if we only called the enqueue function four times and we want to enqueue an element 10, we will succeed.

And it is reasonable that we can not accept more enqueue request because the queue is full now. But what if we dequeue an element? 

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/21/screen-shot-2018-07-21-at-153713.png)

Actually, we should be able to accept one more element in this case.

## Circular Queue

Previously, we have provided a straightforward but inefficient implementation of queue.

A more efficient way is to use a circular queue. Specifically, we may use `a fixed-size array` and `two pointers` to indicate the starting position and the ending position. And the goal is to `reuse the wasted storage` we mentioned previously.

Let's take a look at an example to see how a circular queue works. You should pay attention to the strategy we use to `enqueue` or `dequeue` an element.

## Design

Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO \(First In First Out\) principle and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Your implementation should support following operations:

* `MyCircularQueue(k)`: Constructor, set the size of the queue to be k.
* `Front`: Get the front item from the queue. If the queue is empty, return -1.
* `Rear`: Get the last item from the queue. If the queue is empty, return -1.
* `enQueue(value)`: Insert an element into the circular queue. Return true if the operation is successful.
* `deQueue()`: Delete an element from the circular queue. Return true if the operation is successful.
* `isEmpty()`: Checks whether the circular queue is empty or not.
* `isFull()`: Checks whether the circular queue is full or not.

**Example:**

```text
MyCircularQueue circularQueue = new MyCircularQueue(3); // set the size to be 3
circularQueue.enQueue(1);  // return true
circularQueue.enQueue(2);  // return true
circularQueue.enQueue(3);  // return true
circularQueue.enQueue(4);  // return false, the queue is full
circularQueue.Rear();  // return 3
circularQueue.isFull();  // return true
circularQueue.deQueue();  // return true
circularQueue.enQueue(4);  // return true
circularQueue.Rear();  // return 4
```

**Note:**

* All values will be in the range of \[0, 1000\].
* The number of operations will be in the range of \[1, 1000\].
* Please do not use the built-in Queue library.

```python
class CircularQueue:
    def __init__(self, k):
        global data, head, tail, size
        data = [None] * k
        head = -1
        tail = -1
        size = k

    def enQueue(self, value):
        global head, tail
        if CircularQueue.isFull(self) is True:
            return False

        if CircularQueue.isEmpty(self) is True:
            head = 0

        tail = (tail + 1) % size
        data[tail] = value
        return True

    def deQueue(self):
        global head, tail
        if CircularQueue.isEmpty(self) is True:
            return False

        if head == tail:
            head = -1
            tail = -1
            return True

        head = (head + 1) % size
        return True

    def Front(self):
        if CircularQueue.isEmpty(self) is True:
            return -1

        return data[head]

    def Rear(self):
        if CircularQueue.isEmpty(self) is True:
            return -1

        return data[tail]

    def isEmpty(self):
        return head == -1

    def isFull(self):
        return (tail + 1) % size == head
```

