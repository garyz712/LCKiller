from typing import List

def selectionSort(nums: List[int]) -> None:
        for i in range(len(nums)):
            smallest = i
            for j in range(i+1,len(nums)):
                if nums[j]<nums[smallest]:
                    smallest = j
            nums[i], nums[smallest] = nums[smallest], nums[i]

            
def insertionSort(nums: List[int]) -> None:
        for i in range(1,len(nums)):
            j = i
            while j>0 and nums[j] < nums[j-1]:
                nums[j],nums[j-1] = nums[j-1],nums[j]
                j-=1

def quicksortNaive(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left  = [x for x in arr if x < pivot]
    mid   = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksortNaive(left) + mid + quicksortNaive(right)
                
def partition(l, r, nums):
    # Last element will be the pivot and the first element is the pointer
    pivot, ptr = nums[r], l # pointer is the first element that is larger than the pivot
    for i in range(l, r): # do not include the pivot because it needs to be processed separately
        if nums[i] <= pivot: # move the smaller numbers to the left hand side in their original order
            nums[i], nums[ptr] = nums[ptr], nums[i]
            ptr += 1
    nums[ptr], nums[r] = nums[r], nums[ptr] #last prt cannot increase, move the pivot after the last smaller number 
    return ptr #return the pivot position
 
def quicksort(l, r, nums):
    if len(nums) == 1: 
        return 
    if l < r:
        pi = partition(l, r, nums)
        quicksort(l, pi-1, nums) 
        quicksort(pi+1, r, nums) 




def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
 
    # create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)
 
    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
 
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
 
    # Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray
 
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
 
    # Copy the remaining elements of L[], if there
    # are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
 
    # Copy the remaining elements of R[], if there
    # are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
 
# l is for left index and r is right index of the
# sub-array of arr to be sorted
 
 
def mergeSort(arr, l, r):
    if l < r:
 
        # Same as (l+r)//2, but avoids overflow for
        # large l and h
        m = l+(r-l)//2
 
        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)

def minimumSwaps(popularity):
    n = len(popularity)

    # Step 1: Create a list of tuples (index, value)
    indexed = list(enumerate(popularity))
    print(indexed)

    # Step 2: Sort by value descending, keeping track of original indices
    indexed.sort(key=lambda x: -x[1])
    print(indexed)

    visited = [False] * n
    swaps = 0

    # Step 3: Traverse the array to find cycles
    for i in range(n):
        # If already in the correct position or visited, skip
        if visited[i] or indexed[i][0] == i:
            continue

        # Find the size of the cycle
        cycle_size = 0
        j = i

        while not visited[j]:
            visited[j] = True
            j = indexed[j][0]
            cycle_size += 1

        if cycle_size > 1:
            swaps += (cycle_size - 1)

    return swaps

# Example
popularity = [3, 4, 1, 2]
popularity = [4, 3, 1, 5, 2]
print(minimumSwaps(popularity))  # Output: 2


#merge sort linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeSort(head):
    if not head or not head.next:  # base case
        return head
    
    # Step 1: split into halves
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    mid = slow.next
    slow.next = None  # cut the list
    
    # Step 2: sort both halves
    left = mergeSort(head)
    right = mergeSort(mid)
    
    # Step 3: merge sorted halves
    return merge(left, right)

def merge(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    
    tail.next = l1 or l2  # attach the remainder
    return dummy.next


# O(n) + O(1) cyclic sort: use input array with idx as hash map
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)

    # Use cycle sort to place positive elements smaller than n
    # at the correct index
    i = 0
    while i < n:
        correct_idx = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            # swap
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Iterate through nums
    # return smallest missing positive integer
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # If all elements are at the correct index
    # the smallest missing positive number is n + 1
    return n + 1


# Greedy wiggleSort: Reorder nums such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)-1):
            if i%2==0:
                if nums[i] > nums[i+1]:
                    nums[i] , nums[i+1] = nums[i+1] , nums[i]               
            else:
                if nums[i] < nums[i+1]:
                    nums[i] , nums[i+1] = nums[i+1] , nums[i]
        return nums


# quick select the kth element in unsorted array
import random

def quickselect(arr, k):
    """
    Return the k-th smallest element (0-indexed) in unsorted arr.
    Modifies arr in-place (like real std::nth_element).
    Average time: O(n), Worst: O(n²) → practically never happens never.
    """
    if not arr:
        raise ValueError("array is empty")
    
    def select(left, right, k):
        while True:                              # tail-recursion optimized
            if left == right:
                return arr[left]
            
            # Random pivot → makes worst case astronomically unlikely
            pivot_idx = random.randint(left, right)
            
            # Partition around the pivot (Hoare or Lomuto – both are standard)
            pivot_idx = partition(left, right, pivot_idx)
            
            if k == pivot_idx:
                return arr[k]
            elif k < pivot_idx:
                right = pivot_idx - 1            # search only left part
            else:
                left = pivot_idx + 1             # search only right part
    
    return select(0, len(arr)-1, k)


# Standard Lomuto partition used in CLRS and most textbooks
def partition(left, right, pivot_idx):
    pivot = arr[pivot_idx]
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]   # move pivot to end
    store_idx = left
    for i in range(left, right):
        if arr[i] < pivot:
            arr[i], arr[store_idx] = arr[store_idx], arr[i]
            store_idx += 1
    arr[right], arr[store_idx] = arr[store_idx], arr[right]      # move pivot to final place
    return store_idx