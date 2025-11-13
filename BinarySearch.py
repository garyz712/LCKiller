def searchInsert(nums, target):
    left, right = 0, len(nums)-1
    while left <= right: # Simply check out left==right to understand the left, right pointers meaning
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1 # might be -1 when mid==0, so useless
        else:
            left = mid + 1 # first index where target is greater than everything before
    
    return left # So it is also the insertion position


def search(nums, target):
    """
    Search for target in a rotated sorted array using binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums:
        return -1
    
    left, right = 0, len(nums) - 1

    while left <= right:

        mid = (left+right)//2  #if (left+right)%2 == 0 else (left+right)//2 +1
        if nums[mid]==target:
            return mid
        if nums[mid]>=nums[left]: # >= to avoid infinite loop!
            if nums[left]<= target < nums[mid]: # nums[mid]!=target
                right = mid-1 #If write left = mid or right = mid, mid could remain unchanged, causing an infinite loop. Always increment/decrement past mid.
            else:
                left = mid+1
        else:
            if nums[mid]< target <= nums[right]:
                left = mid+1
            else:
                right = mid-1
    return -1

# search in rotated sorted array with duplicates: O(n) worst case
def search(self, nums: List[int], target: int) -> bool:
    left, right = 0, len(nums)-1

    while left<=right: #need to run the loop when len=1, so use <= instead of <
        mid = (right+left) //2

        if nums[mid] == target:
            return True

        if nums[mid] > nums[left]: #lhs is increasing
            if nums[left]<=target < nums[mid]: #target!=mid for sure
                right = mid-1
            else:
                left = mid+1
        elif nums[mid] < nums[left]: #rhs is non decreasing eg. 3  1  1
            if nums[mid] < target <= nums[right]: #target!=mid for sure
                left = mid+1
            else:
                right = mid-1
        else: # mid==left,     1  1  0 / 1 1 1 
            left = left +1 # turn into linear search

    return False

        

# Test cases
def test_search():
    test_cases = [
        # 1. Typical rotated array, target in right half
        {
            "nums": [4, 5, 6, 7, 0, 1, 2],
            "target": 0,
            "expected": 4
        },
        # 2. Typical rotated array, target in left half
        {
            "nums": [4, 5, 6, 7, 0, 1, 2],
            "target": 5,
            "expected": 1
        },
        # 3. Target not in array
        {
            "nums": [4, 5, 6, 7, 0, 1, 2],
            "target": 3,
            "expected": -1
        },
        # 4. Single element, target present
        {
            "nums": [1],
            "target": 1,
            "expected": 0
        },
        # 5. Single element, target not present
        {
            "nums": [1],
            "target": 0,
            "expected": -1
        },
        # 6. Two elements, rotated
        {
            "nums": [2, 1],
            "target": 1,
            "expected": 1
        },
        # 7. No rotation (sorted array)
        {
            "nums": [1, 2, 3, 4, 5],
            "target": 3,
            "expected": 2
        },
        # 8. No rotation, target not present
        {
            "nums": [1, 2, 3, 4, 5],
            "target": 6,
            "expected": -1
        },
        # 9. Full rotation (back to original)
        {
            "nums": [1, 2, 3],
            "target": 2,
            "expected": 1
        },
        {
            "nums": [1, 2, 3],
            "target": 1,
            "expected": 0
        },
        {
            "nums": [1, 2, 3],
            "target": 3,
            "expected": 2
        },
        # 10. Empty array
        {
            "nums": [],
            "target": 1,
            "expected": -1
        },
        # 11. Rotated array, target at start
        {
            "nums": [3, 4, 5, 1, 2],
            "target": 3,
            "expected": 0
        },
        # 12. Rotated array, target at end
        {
            "nums": [3, 4, 5, 1, 2],
            "target": 2,
            "expected": 4
        },
        # 13. Large numbers within constraints
        {
            "nums": [10000, -10000, 0, 5000],
            "target": -10000,
            "expected": 1
        },
        # 13. Large numbers within constraints
        {
            "nums": [10000, -10000, 0, 5000],
            "target": 10000,
            "expected": 0
        },
        # 13. Large numbers within constraints
        {
            "nums": [10000, -10000, 0, 5000],
            "target": 0,
            "expected": 2
        },
        # 13. Large numbers within constraints
        {
            "nums": [10000, -10000, 0, 5000],
            "target": 5000,
            "expected": 3
        }
    ]

    for i, test in enumerate(test_cases, 1):
        result = search(test["nums"], test["target"])
        assert result == test["expected"], (
            f"Test case {i} failed: nums={test['nums']}, "
            f"target={test['target']}, expected={test['expected']}, got={result}"
        )
        print(f"Test case {i} passed: nums={test['nums']}, target={test['target']}, result={result}")

def find_median_rotated(nums):
    """
    Find the median of a rotated sorted array
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums:
        return None
    
    n = len(nums)
    
    # Step 1: Find the rotation point (index of minimum element)
    def find_min_index():
        left, right = 0, len(nums)-1
        while left < right: 
            mid = (left + right) // 2                 
            if nums[mid] > nums[right]: # cannot be nums[left] because [0,1,2]
                left = mid + 1
            else:
                right = mid
        return right
    
    # Find rotation point
    pivot = find_min_index()
    
    return nums[(pivot+ n//2)%n] if n%2!=0 else (nums[(pivot+ n//2)%n]+nums[(pivot+ n//2)%n - 1])/2

# Test cases
def test_median():
    # Test case 1: Odd length
    nums1 = [4, 5, 6, 7, 1, 2, 3]
    assert find_median_rotated(nums1) == 4, "Test case 1 failed"
    print(f"Median of {nums1}: {find_median_rotated(nums1)}")
    
    # Test case 2: Even length
    nums2 = [3, 4, 1, 2]
    assert find_median_rotated(nums2) == 2.5, "Test case 2 failed"
    print(f"Median of {nums2}: {find_median_rotated(nums2)}")
    
    # Test case 3: No rotation
    nums3 = [1, 2, 3, 4, 5]
    assert find_median_rotated(nums3) == 3, "Test case 3 failed"
    print(f"Median of {nums3}: {find_median_rotated(nums3)}")
    
    # Test case 4: Single element
    nums4 = [1]
    assert find_median_rotated(nums4) == 1, "Test case 4 failed"
    print(f"Median of {nums4}: {find_median_rotated(nums4)}")

    # Test case 5: 
    nums3 = [6, 1, 2, 3, 4, 5]
    assert find_median_rotated(nums3) == 3.5, "Test case 5 failed"
    print(f"Median of {nums3}: {find_median_rotated(nums3)}")
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_search()
    test_median()
    print("All test cases passed!")

# binary search for windows + exclusive evaluation
# In standard binary search with while left < right, the right index is never evaluated as mid inside the for loop. At each iteration: mid = floor((left + right) / 2) → so mid < right. mid==right only possible when left == right, but at that time the answer is already found and the loop already terminates.

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:

        left, right = 0, len(arr) - k  # search the window start index

        while left < right: 
            # mid==right only possible when left == right, but before that the loop has already ended, so mid=right and arr[mid + k] never get evaluated
            mid = (left + right) // 2

            # CANNOT use arr[mid + k -1] because left=mid in that case
            # Use arr[mid+k] instead, but will arr[mid+k] will out of range? NO! Because  #arr[mid+k] will be out of range only when mid = right-k, but this mid will not be evaluated, so you can assume there is always an extra num on the right of the window in while loop
            
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1 # If the element just outside the window on the right (arr[mid + k]) is closer to x than the one on the left edge (arr[mid]), the window isn't optimal — the left pointer must be in mid 's lhs.

            #The current window arr[mid : mid + k] is either better (overall closer to x) or equally good compared to the one starting at mid + 1. So the optimal starting index could be mid or its left.
            else:
                right = mid

        return arr[left:left + k]

#search in answer space
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        while left<right:
            mid = (left+right)//2 #eating speed per hour
            
            time = sum([(i // mid if i % mid == 0 else i // mid + 1) for i in piles])
            
            # alternatively, use ceilling function
            # import math
            # time = sum([math.ceil(i / mid) for i in piles])

            if time > h:
                left = mid+1 #bad answer mid is discarded
            else: #if time==h, mid must be included, so it must use right=mid 
                right = mid
        return right

#search in answer space: smart search space
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        left, right = max(weights), sum(weights) # min is the max of weight, max is the sum of weight -> binary search space
        def finddays(capacity):
            s = 0
            days=1
            for i in range(len(weights)):
                s+=weights[i]
                if s>capacity:
                    s = weights[i]
                    days+=1
            return days

        while left<right:
            mid = (left+right)//2 #current max capacity per day
            if finddays(mid) > days:
                left= mid+1
            else:
                right = mid

        return left
    
class Solution:
    def kthSmallest(self, matrix, k):
        n = len(matrix)
        left, right = matrix[0][0], matrix[-1][-1]
        # When n==1, there is no need to do the while loop
        while left < right: # It always terminates when left == right, Gives a clean return (return left), and Avoids off-by-one errors
            mid = (left + right) // 2
            # Count how many elements <= mid
            count = self.countLessEqual(matrix, mid)
            
            if count < k:
                # If fewer than k elements are <= mid, 
                # we need bigger numbers
                left = mid + 1
            else:
                # If at least k elements are <= mid,
                # k-th smallest could be mid or smaller
                right = mid
        
        return left  # or right, they're the same now


    def countLessEqual(self, matrix, mid):
        n = len(matrix)
        count = 0
        row, col = n - 1, 0  # start bottom-left
        
        while row >= 0 and col < n:
            if matrix[row][col] <= mid:
                count += (row + 1)   # everything above is <= mid too
                col += 1
            else:
                row -= 1
        return count
    

class Solution:
    #2D quasi-binary search for matrix O(n+m) ~ O(sqrt(N)) instead of O(log N^2)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        row, col = 0, len(matrix[0])-1

        while row<len(matrix) and col>=0:
            if matrix[row][col]==target:
                return True
            elif matrix[row][col]>target:
                col-=1
            else:
                row+=1
        return False

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2): #assume A is shorter than B so the left half hold the median when n+m is odd
            A, B = nums2, nums1
        else:
            A, B = nums1, nums2

        # Empty A case implicitly handled by Safeguard
        # if len(A)==0: # handle empty A case
        #     if len(B)%2 == 0:
        #         return (B[len(B)//2] + B[len(B)//2 - 1]) /2
        #     else:
        #         return B[len(B)//2]

        left, right = 0, len(A) # search for partition index (number of elements) in shorter array A, so right can be len(A)

        while left<=right: # search even if len(A) is 0
            midA = (left+right)//2 # search for number of elements in A left half, not the A index
            midB = (len(A)+len(B)+1) //2 - midA #calculate the partition point in B

            # Safeguard for empty left/right half partition in A
            Aleft  = A[midA - 1] if midA > 0 else float("-inf") #if 0 element in A half, Aleft=A[-1]=-inf
            Aright = A[midA]     if midA < len(A) else float("inf") #if 0 element in A right, Aright=A[m]=inf
            Bleft  = B[midB - 1] if midB > 0 else float("-inf") #same for B
            Bright = B[midB]     if midB < len(B) else float("inf") #same for B

            if max(Aleft, Bleft)<= min(Bright, Aright): # if all nums in left half is <= than right half
                if (len(A)+len(B))%2==0: # if nums is even, median is the average
                    return ( max(Aleft, Bleft) + min(Bright, Aright)) /2
                else: # if nums is odd, the median is in left half
                    return max(Aleft, Bleft)
            elif Aleft > Bright: # if lefthalfA is longer, search left in A
                right = midA-1
            elif Bleft > Aright: # if lefthalfB is longer, search right in A
                left = midA+1


#Double ways:
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left, right = 0, x//2+1
        # No need to store the search space [0,1,2,3,4,5,6,7,8,..., x]
        while left<=right:# try this because left<right does not generalize so well        
            mid = (left+right)//2
            if mid*mid==x:
                return mid
            elif mid*mid>x:
                right = mid-1
            else:
                left= mid+1
        # when left==right, if mid^2>x, move right to be a little smaller; elif mid^2<x make left a little larger than the real sqrt, so that is why we need the right pointer
        return right #if no exact match and after (left==right) cycle, return right pointer
    
    def mySqrt1(self, x):
        """
        :type x: int
        :rtype: int
        """
        l, r = 0, x // 2 + 1
        # No need to store all search spaces in list: [0,1,2,3,4,5,6,7,8,...,x]

        while l < r: # attempt to use standard binary search
            mid = (l + r) // 2 # if using this as lower bound, l = mid + 1 must be used to prevent infinite loop: If r = mid - 1 and l = mid for [1,2] -> infinite loop! 2 is never reached!

            if(mid * mid > x): # this mid is a good guy, so keep it 
                r = mid
            
            else: # pretend to be a bad guy, go to the right half without including this mid 
                l = mid + 1 

        #instead of directly searching for the value, we need to search for the one after it since we are moving the l pointer
        return x if x == 0 or x == 1 else l - 1 # return carefully! handle 0 and 1 edge cases
    


# Use bisect_left
# DP is thinking:

# "Where can I append myself to form the best sequence?"

# Binary Search is thinking:

# "Where should I insert myself among all possible subsequences?"

# ✅ Append if I'm bigger than all ends.
# ✅ Replace if I'm smaller, to keep future growth flexible.


# Method 1: Dynamic Programming (O(N²))
# ✅ Simple but slower.
def lengthOfLIS(nums):
    n = len(nums)
    dp = [1] * n #dp[i] = the length of the LIS ending at index i.

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Method 2: Greedy + Binary Search (O(N log N))
# ✅ Harder but much faster.
import bisect

def lengthOfLIS(nums):
    sub = [] #sub[i] = the smallest possible last number of an increasing subsequence (not subarray) of length i+1.

    for num in nums:
        # bisect_left searches in sorted sub array.

        # It finds the first position where sub[idx] >= num.

        # If idx == len(sub) → no element ≥ num, so append and increase the maximum subsequence length.

        # Otherwise → replace sub[idx] = num, while the maximum subsequence length remain the same.

        idx = bisect.bisect_left(sub, num)
        if idx == len(sub):
            sub.append(num)
        else:
            sub[idx] = num

    return len(sub)

# find the frequency of an element in sorted array using Binary Search
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        import bisect
        idx1 = bisect.bisect_left(nums, target)
        # idx2 = bisect.bisect_right(nums, target)
        # return idx2-idx1 > len(nums)//2 #find the frequency
        return idx1 + len(nums)//2< len(nums) and nums[idx1 + len(nums)//2] == target # check if the frequency is greater than len(nums)//2
        
# binary search for the first number smaller than desired + hashmap +list of list of list
class SnapshotArray:
    def __init__(self, length: int):
        self.length = length
        self.sa = [[[0,0]] for i in range(length)] #[snapid, val]
        self.snapid = 0 
        
    def set(self, index: int, val: int) -> None:
        if self.sa[index][-1][0]!=self.snapid:
            self.sa[index].append([self.snapid, val]) 
        elif self.sa[index][-1][1] != val:
            self.sa[index][-1][1] =  val
        #print(self.sa, self.snapid)
        
    def snap(self) -> int:
        currentID = self.snapid
        self.snapid+=1
        return currentID
        
    def get(self, index: int, snap_id: int) -> int:
        l = self.sa[index]
        #print(self.sa, self.snapid)
        found = False
        left, right = 0, len(l)-1
        #[[snapid, val], [snapid, val], [snapid, val] ...]
        res = 0
        while left <= right: # we want to find the first id smaller than the desired id, in this case, desired id can be out of right bound; since res cannot directly be left / right, we must assign the res manually when left==right
            mid = (left + right) // 2
            if l[mid][0] <= snap_id: # when the desired id is <= mid, mid can still be the answer, therefore record mid first; however, if we use left = mid, this is an infinite loop since we are doing //2
                res = mid
                left = mid + 1
            else: # when the desired id is smaller than mid, mid can not be the answer, therefore decrease
                right = mid - 1
        return l[res][1] #when the loop ends, you only know res was calulated when mid is smaller than desired for the last time

        # while left < right: # cannot use <= because there exist condition that the id is not found and you have to refer to the previou one
        #     mid = (left+right)//2
        #     if l[mid][0]<snap_id:
        #         left=mid+1 
        #     else:
        #         right = mid
        # #right might be the answer, OR the first id greater than the desired id (if the desired id is in the list), OR it may be the right bound (desired id may be out of bound)
        # return l[right][1] if l[right][0]<=snap_id else l[right-1][1]


# binary search on tuple for finding the largest element <= timestamp in l
class TimeMap:
    def __init__(self):
        self.timemap = collections.defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.timemap[key].append((timestamp, value))
        

    def get(self, key: str, timestamp: int) -> str:
        l = self.timemap[key]
        if not l or l[0][0] > timestamp:
            return ""

        # idx = bisect.bisect_right(arr, target)  # first > target

        left, right = 0, len(l)-1
        result = -1
        # find the largest element <= timestamp in l
        while left <= right: #candidate might need manual updates when left=right
            mid = (left+right) //2 # compute left mid
            if l[mid][0] <= timestamp:
                result = mid        # candidate
                left = mid + 1 # try to find larger
            else:
                right = mid - 1
        #[1, 2, 3, 4, 5],  3.5
        #[1, 2], 2
        #[1], 1

        # Option 2: cause infinite loop because when left= mid -> does not move forward
        # while left < right:
        #     mid = (left+right) //2 # compute left mid
        #     if l[mid] <= timestamp:
        #         left = mid 
        #     else:
        #         right = mid - 1
        
        # Option 3: find the smallest number that is > target than -1
        # while left < right:
        #     mid = (left + right) // 2
        #     if self.key_time_map[key][mid][0] <= timestamp:
        #         left = mid + 1
        #     else:
        #         right = mid

        # # If iterator points to first element it means, no time <= timestamp exists.
        # return "" if right == 0 else self.key_time_map[key][right - 1][1]


        return self.timemap[key][result][1]







# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)



# bisect practice
# bisect_left(a, x):
# Returns the leftmost index where x can be inserted in sorted list a to maintain order.

# If x already exists, you get the index of the first occurrence.

# bisect_right(a, x) (alias bisect.bisect):
# Returns the rightmost index where x can be inserted to maintain order.

# If x already exists, you get the index after the last occurrence.
# bisect_left(a, x) → returns the index of the first element ≥ x
# bisect_right(a, x) → returns the index of the first element > x

class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        ans = []
        potions.sort()

        def binarysearch(spl):
            left, right = 0, len(potions)-1

            while left<right:
                mid = (left+right)//2
                if potions[mid] * spl < success:
                    left= mid+1
                else:
                    right=mid
            return right if potions[right]*spl>=success else -1

        for spell in spells:
            idx = bisect.bisect_left(potions, success/spell)
            ans.append(len(potions) - idx)# if idx ==0, all numbers in potions are successful; elif idx==len(potions), 0 number is successful 

            #idx = binarysearch(spell)
            
            # if idx==-1:
            #     ans.append(0)
            # else:
            #     ans.append(len(potions) - idx)
        return ans
        


# Unsorted Binary Search, search in the answer space for number count
# find the first element in the nums where number of number smaller than this element is greater 
class Solution:
    def findDuplicate1(self, nums: List[int]) -> int:
        left, right = 1, len(nums)-1
        while left<right:
            mid = (left+right)//2
            if mid >= sum([mid>=i for i in nums]): #count of number that is smaller than or equal to mid
                left = mid + 1
            else:
                right = mid
        return right

    # Two pointer Floyd algorithm
    def findDuplicate(self, nums):
        # Find the intersection point of the two runners.
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # Find the "entrance" to the cycle.
        slow2 = nums[0]
        while slow != slow2:
            slow = nums[slow]
            slow2 = nums[slow2]
        
        return slow

# unsorted binary search + assume the array is ascending -> descending at some point (only one peak)
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left<right:
            mid = (left+right)//2
            if nums[mid]>nums[mid+1]:
                right=mid
            else:
                left=mid+1
        return left


# non standard binary search with == vs != 
class Solution:
    def missingNumber(self, arr: List[int]) -> int:
        n = len(arr)
        diff = (arr[-1] - arr[0]) // n
        left, right = 0, n - 1

        while left < right:
            mid = (left + right) // 2
            if arr[mid] == arr[0] + mid * diff:
                left = mid + 1
            else:
                right = mid

        # [100,300,400]
        return arr[0] + left * diff