import heapq
class Solution: # we can simply find the most-used time slot and count the meeting in that time slot, which will be the minimum rooms needed. so we can find the earliest non-started meeting and compare it with the earliest ending meeting, and see if they can use the same room and that is why we have to sort and use min heap

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x:x[0]) #sort the interval by starting time
        heap = [] # use the min heap to store the total room used, and to find the current earliest ending time, and check if the next meeting start after the earliest ending time
        for meeting in intervals:
            if heap==[] or meeting[0]<heap[0]: #if heap is empty or new meeting start time is before the earliest end time in the heap, we need another room
                heapq.heappush(heap,meeting[1])
            else: #if new meeting start after the earliest ending time, we use the same room and also pop the ended meeting and replace it with the most recent meeting
                heapq.heappop(heap)
                heapq.heappush(heap,meeting[1])
        return len(heap) # this is the Maximum rooms EVER needed, because we did not pop all ended meetings, we pop at most once


# Two pointers on two sorted arrays
def minMeetingRooms_twoPointers(intervals):
    """
    Approach 3: Two Pointers (Separate Sorted Arrays)
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return 0
    
    # Separate and sort start times and end times
    starts = sorted([interval[0] for interval in intervals])
    ends = sorted([interval[1] for interval in intervals])
    
    rooms_needed = 0
    max_rooms = 0
    start_ptr = 0
    end_ptr = 0
    
    while start_ptr < len(starts):
        if starts[start_ptr] < ends[end_ptr]:
            # Meeting starts, need a room
            rooms_needed += 1
            max_rooms = max(max_rooms, rooms_needed)
            start_ptr += 1
        else:
            # Meeting ends, free up a room
            rooms_needed -= 1
            end_ptr += 1
    
    return max_rooms

    
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()  # Maintains order of key usage, but you must explicitly call move_to_end(key) to reflect recent use, especially for LRU Cache.
        #If you skip that, OrderedDict will behave like a regular dict with preserved insertion order — not usage order.

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1  # Return -1 if the key is not in the cache
        self.cache.move_to_end(key)  # Mark key as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)  # Update existing key as recently used
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  

# Usage example:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key, value)


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object): # linkedlist adder with carry
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        l3 = ListNode()
        current = l3
        carry = 0 
        while l1 or l2 or carry: #as long as one number is valid, the addition must go on
            l1v = l1.val if l1 else 0
            l2v = l2.val if l2 else 0

            current.val = (l1v + l2v + carry)%10
            carry = (l1v + l2v + carry)//10
            l1 = l1.next if l1 else l1
            l2 = l2.next if l2 else l2
            if l1 or l2 or carry: # if anything is valid next cycle, create a new node, waiting for its value
                current.next = ListNode()
                current = current.next

        return l3


class Solution:
    def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        if not A or not B or not A[0] or not B[0]:
            return []

        m, n, p = len(A), len(A[0]), len(B[0])
        result = [[0] * p for _ in range(m)]

        # Preprocess B: store non-zero values in each row of B
        b_map = {}
        for k in range(len(B)):
            b_map[k] = {}
            for j in range(len(B[0])):
                if B[k][j] != 0:
                    b_map[k][j] = B[k][j]

        # Multiply A and B efficiently
        for i in range(m):
            for k in range(n):
                if A[i][k] != 0:
                    for j in b_map.get(k, {}):
                        result[i][j] += A[i][k] * b_map[k][j]

        return result

        