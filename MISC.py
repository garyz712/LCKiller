class TrieNode:
    def __init__(self):
        self.children = {}  # Mapping from character to TrieNode
        self.is_end = False  # Marks if this node represents the end of a word

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node=self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node=self.root
        for c in word:
            if c not in node.children:
                return False               
            node = node.children[c]       
        return node.is_end #return True if node.is_end else False
        
    def startsWith(self, prefix: str) -> bool:
        node=self.root
        for c in prefix:
            if c not in node.children:
                return False               
            node = node.children[c]       
        return True #return True if find a prefix
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)



# prefix & suffix
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res=[1] * len(nums) #res array is O(n) but only O(1) extra space

        for i in range(1, len(nums)):#need to store all prefix product for backward pass, they can't be computed on the fly in backward pass, since nums[i] need to be reused, you can't modify nums at this time
            res[i] = res[i-1] * nums[i-1] #res is the prefix product array

        curSuffixProduct=1 # O(1) extra space: you no longer need to store the suffix array in nums since each entry will be used only once and can be computed on the fly
        for j in range(len(nums)-1, -1, -1):
            res[j] *= curSuffixProduct
            curSuffixProduct *= nums[j] #update curSuffixProduct for next res
        return res
    


        

class Solution: # Boyer-Moore Voting Algorithm
    def majorityElement(self, nums: List[int]) -> int:
        cadidate=None
        count=0 #the support of current candidate vs opposition
        for i in nums:
            if count==0:  #if two candidate count is the same count==0: whoever won the next vote is the new candidate
                candidate = i
            if i!=candidate:
                count-=1
            else:
                count+=1
        return candidate

# _dontcare,_,_,_,_,_,candidate,c,c,c,c,c,c,c,c


#rotation is equivalent to 3 reversion
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)

        k %= n  # Rotating an array by k steps is equivalent to rotating it by k % n steps.
        nums[:n-k] = reversed(nums[:n-k])
        nums[n-k:] = reversed(nums[n-k:])
        nums[:]= reversed(nums) #in place modification, no return or rebind is allowed
    
    # 4,3,2,1,7,6,5
    # 5,6,7,1,2,3,4


# interval question
class Solution(object):# three while loop with O(n) time
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        res=[]
        i=0
        # Add intervals before newInterval
        while i<len(intervals) and intervals[i][1]<newInterval[0] :
            res.append(intervals[i])
            i+=1
        
        # Merge overlaps with newInterval
        while i<len(intervals) and intervals[i][0] <= newInterval[1]:  
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i+=1

        res.append(newInterval)

        # Add remaining intervals
        while i<len(intervals):
            res.append(intervals[i])
            i+=1
        return res

        
class Solution: # reverse a 32 bit unsigned int
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n): # left and right shift and & | operator
        # for _ in range(32):
        #     result = (result << 1) | (n & 1)
        #     n >>= 1
        # return result

        return int(bin(n)[2:].zfill(32)[::-1], 2)
        

# Divide and Conquer + BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mid = len(nums)//2
        node = TreeNode(nums[mid]) #creat the root node
        node.left = self.sortedArrayToBST(nums[:mid]) # divide nums into half
        node.right = self.sortedArrayToBST(nums[mid+1:]) # and conquer by setting its left and right node
        return node #return the current root


# Bit operation
class Solution:
    def hammingWeight(self, n: int) -> int:
        #return sum([int(c) for c in str(bin(n))[2:]])

        sum = 0
        while (n != 0): #eventually n will be 0 after shifting
            sum+=n &1 # take the last bit
            n >>= 1 # unsigned shift left one bit -> remove the last bit
        
        return sum
        
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for shift in range(32):
            b = sum([(nums[i]>> shift)&1 for i in range(len(nums))])%3
            res += (b << shift)
        if res >= 2 ** 31: # convert to negative if num out of range
            res -= 2 ** 32
        return res
    
# Brian Kernighan's Algorithm
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        # res = left
        # for i in range(left+1, right+1):
        #     res = i & res
        #     if res ==0:
        #         return 0
        # return res
        
        while left < right: # keep turning off the rightmost "1" bit until right is smaller than left, at this point, right will be the common prefix for ALL numbers in the range!
            right &= (right - 1) # whenever you do this, you change the rightmost "1" bit to "0" bit in right
        return right # this is the common prefix for ALL numbers in the range, it is also the answer -> AND all numbers!
        
#trie prefix tree + dfs + backtracking
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None
        #don't know its own char

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        curr = root

        #buid the trie prefix tree
        for word in words:
            for c in word:
                if c not in curr.children:
                    curr.children[c] = TrieNode() #store the current char as a TrieNode
                    curr = curr.children[c]
                else:
                    curr = curr.children[c]
            curr.word = word #store the whole word
            #print("stored",word)
            curr = root

        #dfs on the board
        def dfs(i, j, node):
            c = board[i][j]
            if c in node.children:
                #print(c, "current char in keys")
                if node.children[c].word!= None:
                    res.append(node.children[c].word)
                    node.children[c].word = None #avoid duplicate

                node = node.children[c]

                board[i][j] = "-" #The same letter cell may not be used more than once in a word.                
                for x,y in [(0,1),(0,-1),(1,0),(-1,0)]:
                    if 0<=i+x<len(board) and 0<=j+y<len(board[0]):
                        dfs(i+x, j+y, node)

                board[i][j] = c #backtrack

            else:
                return

        res = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(i, j, root)

        return res

        
#hash map in hash map / create n hashmap for n iterations
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        from collections import defaultdict
        
        res = 1
        for i in range(len(points)):
            count = defaultdict(int)
            for j in range(i+1, len(points)):
                if points[j][0] - points[i][0] == 0:
                    slope = float("inf")
                else:
                    slope = (points[j][1] - points[i][1]) /(points[j][0] - points[i][0]) 
                count[slope] += 1

                res = max(res, count[slope]+1)

        return res

        