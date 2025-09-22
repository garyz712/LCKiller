#recursion with space
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==1 or n==1:
            return x
        if n==0:
            return 1
        if n<0:
            return 1/self.myPow(x, -n)
        
        if n%2==0:
            half = self.myPow(x, n/2) #must be saved to save time
            #1+2+4+...+2^logn = 2^log(n+1)-1 = O(n) calls
            return half*half #from O(n) -> O(logn) calls using O(logn) memory
        else:
            return self.myPow(x, n-1)*self.myPow(x, 1)
class Solution:
    def subsetsRecur(self, nums: List[int]) -> List[List[int]]:
        ans= []
        def dfs(s, i):
            if i==len(nums):
                ans.append(s) #must append a copy because s will be modified later
                return

            dfs(s[:], i+1) #same s as below
            s.append(nums[i]) 
            dfs(s[:], i+1)

        dfs([], 0)
        return ans

    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans= []
        subset = []#if using backtrack, no need to pass this as copy
        def dfs(i):
            if i==len(nums):
                ans.append(subset[:]) #must append a copy because s will be modified later
                #print(ans)
                return

            dfs(i+1) #same s as below
            subset.append(nums[i]) 
            dfs(i+1)
            subset.pop() # must use backtrack to pop from s because this s is the same as the previously passed s
        dfs(0)
        return ans
        
        
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res=[]
        def dfs(start, path, total):
            if total == target:
                res.append(path)
                return
            elif total > target:
                return
            
            for i in range(start, len(candidates)):
                dfs(i, path+[candidates[i]], total+candidates[i])
        dfs(0, [], 0)
        return res
    
    # def backtrack(start, path, total):
    #     if total == target:
    #         res.append(path[:])
    #         return
    #     for i in range(start, len(nums)):
    #         if total + nums[i] <= target:
    #             path.append(nums[i])
    #             backtrack(i, path, total + nums[i])
    #             path.pop()  # 🧠 undo


# DFS with space vs backtracking solution
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution: 
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        # If it's a leaf, check if the current value equals the remaining target
        if not root.left and not root.right:
            return targetSum == root.val

        # Recurse on left and right with updated target
        return (
            self.hasPathSum(root.left, targetSum - root.val) or
            self.hasPathSum(root.right, targetSum - root.val)
        )

    # def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    #     ans = False
    #     s = 0 
    #     def dfs(node):
    #         nonlocal s, ans
    #         if node == None:
    #             return 
    #         s += node.val

    #         if node.left == None and node.right == None:               
    #             ans |= targetSum == s 

    #         dfs(node.right) 
    #         dfs(node.left)
    #         s -= node.val #backtrack with O(1) memory

    #     dfs(root)
    #     return ans 

    
        
         
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(numsLeft, perm):
            if len(numsLeft)==0:               
                res.append(perm)
                return
            for num in numsLeft:
                dfs([x for x in numsLeft if x != num], perm+[num])
        dfs(nums, [])
        return res

        # res = []
        # used = [False] * len(nums)
        # def backtrack(path):
        #     if len(path) == len(nums):
        #         res.append(path.copy())  #append a copy
        #         return
        #     for i in range(len(nums)):
        #         if not used[i]:
        #             used[i] = True
        #             path.append(nums[i])
        #             backtrack(path)
        #             path.pop()
        #             used[i] = False
        # backtrack([])
        # return res
                

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res=[]

        def dfs(num, newK, comb):
            if newK==0:
                res.append(comb)
                return
            for i in range(num, n+1):
                dfs(i+1,newK-1,comb+[i])


        # def backtrack(start, path): 
        #     if len(path)==k:
        #         print(path)
        #         res.append(path.copy()) #append a copy
        #         return 
        #     for i in range(start, n+1):
        #         path.append(i)
        #         backtrack(i+1, path)
        #         path.pop()

        dfs(1,k,[])
        #backtrack(1, [])

        return res

#backtracking by modifying & reverting global variable board
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        res = False
        n, m = len(board), len(board[0])
        def dfs(curX, curY, last, l):
            if l == len(word) and l:
                nonlocal res #must use nonlocal to re-assign variable outside
                res = True
                return 
            if 0<=curX<n and 0<=curY<m:
                if board[curX][curY] == word[l]:
                    board[curX][curY] = "#" #marked as visited so it won't be used twice in this path
                    #print(word[l],l)
                    directions = [[0,1],[1,0],[-1,0],[0,-1]] 
                    for d in directions:
                        if d!=[-last[0], -last[1]]:
                            dfs(curX+d[0], curY+d[1], d, l+1)
                            if res:
                                return #if find a solution, return earlier
                    board[curX][curY] = word[l] #after finish exploring this path, backtrack!
        for i in range(n):
            for j in range(m):
                dfs(i, j, [0,0], 0)
                if res:
                    return True
        return False

#backtrack in multiple sets vs backtrack using the original graph
class Solution:
    def totalNQueens(self, n: int) -> int:
        res = 0
        def backtrack(row):
            if row == n:
                nonlocal res
                res+=1
                return
            for col in range(n):
                if col not in cols and row-col not in posdiag and row+col not in negdiag:
                    cols.add(col)
                    posdiag.add(row-col)
                    negdiag.add(row+col)
                    backtrack(row+1) 
                    negdiag.remove(row+col)
                    posdiag.remove(row-col)
                    cols.remove(col)  

        cols=set()
        posdiag = set() #row-col=constant
        negdiag = set() #row+col=constant
        backtrack(0)
        return res

        # def is_valid(board, row, col):
        #     for i in range(row):
        #         if board[i][col] == 'Q':
        #             return False
        #         if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
        #             return False
        #         if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
        #             return False
        #     return True

        # def backtrack(row):
        #     nonlocal count
        #     if row == n:
        #         count += 1
        #         return
        #     for col in range(n):
        #         if is_valid(board, row, col):
        #             board[row][col] = 'Q'
        #             backtrack(row + 1)
        #             board[row][col] = '.'

        # board = [['.'] * n for _ in range(n)]
        # count = 0
        # backtrack(0)
        # return count
