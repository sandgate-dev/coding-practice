# queue\_time

There is a queue for the self-checkout tills at the supermarket. Your task is write a function to calculate the total time required for all the customers to check out!

The function has two input variables:

* customers: an array \(list in python\) of positive integers representing the

  queue. Each integer represents a customer, and its value is the amount of

  time they require to check out.

* n: a positive integer, the number of checkout tills.

The function should return an integer, the total time required.

EDIT: A lot of people have been confused in the comments. To try to prevent any more confusion:

* There is only ONE queue, and
* The order of the queue NEVER changes, and
* Assume that the front person in the queue \(i.e. the first element in the

  array/list\) proceeds to a till as soon as it becomes free.

* The diagram on the [wiki page](https://en.wikipedia.org/wiki/Thread_pool) may be useful. 

So, for example:

`queue_time([5,3,4], 1)` should return 12 because when n=1, the total time is just the sum of the times

`queue_time([10,2,3,3], 2)` should return 10 because here n=2 and the 2nd, 3rd, and 4th people in the queue finish before the 1st person has finished.

`queue_time([2,3,10], 2)` should return 12

```python
def queue_time(customers, n):
    aisles = [0] * n
    for i in customers:
        aisles.sort()
        aisles[0] += i
    return max(aisles)

customers = [41,16,13,11,4,13,30,25,47,39,45,43,50,16,40,3,4,47,40,32]
print(queue_time(customers, 4))  # 155
```

