class Solution: # classic sliding window / two pointers problem
    def lengthOfLongestSubstring(self, s: str) -> int:
        d = {}
        max_len = 0
        left, right=0,0
        for i in range(len(s)):
            if  s[i] in d and d[s[i]]>=left:
                left = d[s[i]]+1 #only update left pointer when we see repeated char after the current left pointer, if it is before left, we don't care
                #so this update is shrink only, and only shrink window when we found repeated char in the latest window
            d[s[i]]=i # update the latest char position
            right=i
            max_len = max(right-left+1, max_len) # current max length without repetition
        
        return max_len
    
# The array only has positive integers →
# ✅ Once the sum exceeds target, moving the left boundary right will only decrease the sum.

# So you can expand right pointer, and shrink left pointer greedily.

# This is a classic two pointers (sliding window) situation.

# Sliding window is perfect when:

# You have positive numbers

# You need to find contiguous subarrays

# You want minimal/maximal length

class Solution: #may use bisect_left for O(nlogn) solution
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        ans, left,s=10**6,0,0

        for right in range(len(nums)):
            s+=nums[right]                   
            while s>=target:
                ans=min(ans, right-left+1)
                s-=nums[left]
                left+=1
                
        return ans if ans!=10**6 else 0
    
#Two pointers start at both ends
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1 #two pointers at both ends, so width must be decreasing -> height must be increasing to find the max area
        ans = (right-left)*min(height[left], height[right])

        for i in range(len(height)):
        #if you move the longer side, the area must be decreasing since the area is decided by the shorter side, so you have to move the shorter side
            if height[left]<height[right]:
                left+=1
            else:
                right-=1
                
            area = (right-left)*min(height[left], height[right])
            if area>ans:
                ans = area
        return ans


#Inverse thinking : two pointer for swap
class Solution:

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        # left will find the first zero for swap in the window, right is the first non-zero during process

        for right in range(len(nums)):
            if nums[right] != 0: #keep increasing both left and right when no zero until a zero is found
                if left != right:
                    nums[left], nums[right] = nums[right], nums[left]
                left+=1



#sort & fix pointer & move the other two pointers &check duplicate through sorted traversal

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # skip duplicates of nums[i]
                
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                
                if s == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    # Skip duplicates
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif s < 0:
                    left += 1
                else:
                    right -= 1
                    
        return ans


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans=[]
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # since if i-1 is checked, and nums[i]==nums[i-1], everything after nums[i] is also checked

            left, right = i+1, len(nums)-1 #search RHS of the first pointer, LHS is already searched
            for j in range(i+1,len(nums)-1): #-1 because left and right cannot be the same position 
                if left >= right:
                        break
                s=nums[left]+nums[right]+nums[i]
                if s==0:
                    ans.append([nums[left],nums[right],nums[i]])
                    left += 1 #check the next triplet
                    right -= 1 #check the next triplet
                    #check duplicate in sorted array
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif s>0:
                    right-=1
                else:
                    left+=1

        return ans
#[-4, -1, -1,0,1,2]
        
# counter / dictionary + for + while loop
from collections import Counter
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s: #empty edge case
            return ""
        
        t_counter = Counter(t) #dict counter for t frequency
        window = {} #substring window frequency dict       
        have, uniqueNeeded = 0, len(t_counter) # instead of checking if each substring contains all t char, we count how many unique char in t do we have in substring

        ans = s
        left = 0
        foundOne = False
        
        for right in range(len(s)): #use for loop instead of while to avoid right+=1
            window[s[right]] = window.get(s[right], 0) + 1 #add/update window freq dict            
            # if the new char in t and both freqs of it exactly match
            if s[right] in t_counter and window[s[right]] == t_counter[s[right]]:
                #another unique characters in t meet its count in the window
                have += 1 
                               
            while have == uniqueNeeded:#when window contain all t
                foundOne = True # we found one matching substring

                # Update result when new window is smaller than curr
                if (right - left + 1) < len(ans): 
                    ans = s[left: right+1]
                
                # Pop ONE left from the window dict
                window[s[left]] -= 1

                # if the removed char in t and its new freq < t required
                if s[left] in t_counter and window[s[left]] < t_counter[s[left]]:
                    # we lose one char in our list
                    have -= 1

                left += 1 # add left pointer until window does not contains all t
    
        return ans if foundOne else "" #if no eligible substring, return empty


class Solution:
    
    # two pointer, no swap
    def removeDuplicates(self, nums: List[int]) -> int:
        size = len(nums)
        insertIndex = 1
        for i in range(1, size):
            # Found unique element
            if nums[i - 1] != nums[i]:
                # Insert the number only at its first appearance
                nums[insertIndex] = nums[i]
                # Incrementing insertIndex count by 1
                insertIndex = insertIndex + 1
        return insertIndex

        # two pointer, swap
    def removeDuplicates1(self, nums: List[int]) -> int:
        last = 0
        for i in range(1,len(nums)):
            if nums[i] != nums[last]:
                nums[last+1], nums[i] =  nums[i], nums[last+1]
                last+=1
        return last+1

    #two pointer with counter, no swap
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        insertpos = 1
        count = 1 #count the number of times a number has occurred
        for i in range(1,len(nums)):
            if nums[i]==nums[i-1] and count<2: #the only way to decide if this is a old number, and if this number has occured for less than twice, replace/swap the number, increase the count, increase the insertpos, else pass
                count+=1 #if it is old, increase the counter
                nums[insertpos] = nums[i]
                insertpos+=1
            elif nums[i]!=nums[i-1]: #if it is new, reinitialize the counter, replace/swap the number, increase the insertpos
                count=1
                nums[insertpos] = nums[i]
                insertpos+=1
                   
        return insertpos
    
# 1,1,2,2,2/,3,3,4,5,5,5,5,6
# 1,1,2,2,3,3,3/,4,5,5,5,5,6
# 1,1,2,2,3,3,4,4/,5,5,5,5,6
# 1,1,2,2,3,3,4,5,5,5/,5,5,6
# 1,1,2,2,3,3,4,5,5,6,5,5,6/



class Solution:
    #Two pointer solution
    def trap2(self, height: List[int]) -> int:
        if len(height)<3:
            return 0
        total = 0
        r, l = len(height)-1, 0
        while l+1< len(height) and height[l]<=height[l+1]:
            l+=1
        while r-1>=0 and height[r]<=height[r-1]:
            r-=1
        slow, fast = l, l
        #print(l,r)
        while fast<r:
            while height[fast]>height[fast+1]:          
                fast+=1
            m = max(height[fast:r+1])
            if m>=height[slow]:
                while (fast+1< len(height) and height[fast]<=height[fast+1]) or height[fast]<height[slow]:          
                    fast+=1 #find the first value that starts to decrease and it is greater or equal than slow
                cap = height[slow]
            else:
                while height[fast]!=m:          
                    fast+=1 #find the biggest value that is smaller than slow
                cap = height[fast]

            #print(slow, fast, cap)
            slow+=1
            while slow<fast:
                if cap - height[slow]>0:      
                    total += cap - height[slow]
                    #print(cap - height[slow])
                slow+=1
                
                

        return total
    # "four" pointer solution: 2 traversal points + 2 reference pointers
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        l, r = 0, len(height) - 1  # Two pointers
        left_max, right_max = 0, 0  # Max seen so far from both ends
        total = 0

        while l < r:
            if height[l] < height[r]:
                if height[l] >= left_max:
                    left_max = height[l]  # Update left bound
                else:
                    total += left_max - height[l]  # Trap water
                l += 1
            else:
                if height[r] >= right_max:
                    right_max = height[r]  # Update right bound
                else:
                    total += right_max - height[r]  # Trap water
                r -= 1

        return total


#dictionary in sliding window, stepwise sliding
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import defaultdict
        res=[]
        countref = defaultdict(int)
        for word in words:
            countref[word] += 1
        length = len(words[0])

        for i in range(length):
            count = defaultdict(int) #need a counter dict to track the words in sliding window

            for j in range(i, len(s)-length+1, length):
                #print(j)
                if s[j:j+length] in words:
                    #print(s[j:j+length])
                    count[s[j:j+length]] += 1

                if j - length * len(words) >=0:
                    start = s[j - length * len(words): j - length * (len(words)-1)]
                    if start in words:
                        count[start]-=1
                if count == countref:               
                    res.append(j-length * (len(words)-1))

        return res

                

# decreasing deque storing idx + sliding window
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        queue = collections.deque() #must store idx instead of values because you have to know where is the queue[0] comming from to decide if it is still in bound
        for i in range(len(nums)):
            if queue and queue[0] == i-k: # if the leftmost queue is out of bound
                queue.popleft() #remove the leftmost element

            #pop queue to ensure it is decreasing
            while queue and nums[queue[-1]]<nums[i]:
                queue.pop()
            
            queue.append(i) #put the current number idx into the queue
           
            if i>=k-1:
                ans.append(nums[queue[0]]) #adding the answer (the left most num in queue)
            #print(queue)
        return ans


# Two pointers two traversals
class Solution:
    def getIntersectionNodeExp(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        lA, lB = 0, 0
        curA, curB = headA, headB

        while curA:
            lA+=1
            curA = curA.next
        while curB:
            lB+=1
            curB = curB.next

        Alead = lA - lB
        curA, curB = headA, headB
        if Alead >=0:
            for i in range(Alead):
                curA = curA.next
        else:
            for i in range(-Alead):
                curB = curB.next

        while curA != curB:
            curA = curA.next
            curB = curB.next
        return curA

# pA pA pA pA->pB pB  pB pB pB pB 
# pB pB pB pB  pB pB->pA pA pA pA
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        pA = headA
        pB = headB

        while pA != pB:
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next

        return pA