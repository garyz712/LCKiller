#1D DP O(n)->O(1) space! Two options at each step
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[-1]

def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev1 = nums[0]
    prev2 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev2, prev1 + nums[i])
        prev1 = prev2
        prev2 = curr

    return prev2

#multi var DP + Kadane algo
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        currMaxProd = 1
        currMinProd = 1
        globProd = -float("inf")
        
        for n in nums:
            temp = max(max(currMaxProd*n, currMinProd*n), n)
            currMinProd = min(min(currMaxProd*n, currMinProd*n), n) #need to save the current minimum because if they are negative and the current number is negative as well, they can also contribute to the maximum product. 
            currMaxProd = temp
            globProd = max(globProd, currMaxProd)
        return globProd


#multi var DP + Kadane algo
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        globMax, globMin = nums[0], nums[0]
        curMax, curMin = 0, 0

        total = 0

        for n in nums:
            curMax = max(curMax+n, n) #the maximum sum including the current n
            curMin = min(curMin+n, n) #the minimum sum including the current n
            total+=n
            globMax = max(globMax, curMax)
            globMin = min(globMin, curMin)

        return max(globMax, total-globMin) if globMax>0 else globMax #if all nums are negative, total-globMin will be 0>globMax -> wrong! -> globMax


        
# 1D DP: Why two loops?
# Because at each amount i, you have MULTIPLE ways to reach it, depending on coin denominations.

def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1) # other amounts cannot be made at this time.
    dp[0] = 0  # 0 coins needed for amount 0, we know amount 0 can be made!

    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1) #minimum number of coins combination using different coins at this step

    return dp[amount] if dp[amount] != float('inf') else -1

# 1D DP: time O(N*sqrt(N)), space O(N), two loops to search for every single possible sqaure number
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [i for i in range(n+1)]

        for i in range(1, len(dp)):
            j=1
            while j**2<=i:
                dp[i] = min(dp[i], 1+dp[i-j**2])
                j+=1
        return dp[-1]

# Bottom up 2D->1D DP: triangle problem
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = triangle[-1] #copy the last row
        for i in range(len(triangle)-2,-1,-1): #so traversal start from the second last row
            for j in range(len(triangle[-1]) - (len(triangle)-1-i)):
                dp[j]= min(dp[j],dp[j+1]) + triangle[i][j]
                
        print("start testing python range function: They both should output 0-2")
        for i in range(0, 3):
            print(i)
        for i in range(2,-1,-1):
            print(i)

        return dp[0]
    
# Binary branching, string dynamic programming: Decode Way
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = [0]*(len(s)+1)
        dp[0]=1 #1 as starting point, because string will not be empty
        dp[1]= 1 if s[0]!="0" else 0 #if string start from 0, there is no way to decode
        for i in range(2, len(s)+1):
            if s[i-1]!="0": #if current char is not 0, number of additional decode ways is the same as previous char
                dp[i]+=dp[i-1]
            if s[i-2]!="0" and 1<=int(s[i-2:i])<=26: #if current double char is 10-26, number of additional ways is the same as two char ahead; 
                dp[i]+=dp[i-2]  #if current double char is 00-09, it is not a valid double char, do not add anything
        return dp[-1]
    
#2D->1D DP based on True / False        
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        n,m,s = len(s1),len(s2),len(s3)
        dp=[[False]* (m+1) for _ in range(n+1)]
        dp[0][0]=True #s1 = "", s2 = "", s3 = ""
        if n + m != s:
            return False

        for i in range(1, n+1): # s2 = ""
            if s1[i-1]==s3[i-1] and dp[i-1][0]:
                dp[i][0]= True
        for i in range(1, m+1): # s1 = ""
            if s2[i-1]==s3[i-1] and dp[0][i-1]:
                dp[0][i] = True

        for i in range(1, n+1):
            for j in range(1, m+1):
                if (s2[j-1]==s3[i+j-1] and dp[i][j-1]) or (s1[i-1]==s3[i+j-1] and dp[i-1][j]): # look at up and left, if the current s3 char is the same as the s1[i-1] or s2[j-1]
                    dp[i][j] = True
        return dp[n][m]
    
    def isInterleave1D(self, s1: str, s2: str, s3: str) -> bool:
        n, m, s = len(s1), len(s2), len(s3)
        if n + m != s:
            return False

        dp = [False] * (m + 1)
        dp[0] = True

        for j in range(1, m + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]

        for i in range(1, n + 1):
            dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
            for j in range(1, m + 1):
                dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or (dp[j - 1] and s2[j - 1] == s3[i + j - 1])

        return dp[m]
        
class Solution: # O(n^3) -> O(n^2) 2D DP + MULTIPLE->TWO options at each step (because only the max matters)!
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        dp=[[0]*n for i in range(k+1)] #max profit at jth day after at most i transactions (k max here)
        for i in range(1, k+1):
            max_diff = -prices[0] + 0 #The maximum value of dp[i-1][b] - prices[b] for all b < j
            for j in range(1, n):
                #option 1. O(n^3) 
                #dp[i][j] = dp[i][j-1]  # default: sell before day j+1 and do nothing on day j+1, take the max profit at jth day
                #for b in range(j):  # Then try all buy days before day j+1 (at most day j -> idx=j-1) and sell on day j+1 (idx=j), and add the max profit at day b-1 from i-1 transactions. However, if you buy on day 1, there is no day 0. Since the question is asking at most k transactions, it is ok to buy and sell on the same day and it will just waste one transaction if you do that, so let' go back to the max profit at day b from i-1 transactions -> dp[i-1][b]
                    #dp[i][j] = max(dp[i][j], prices[j] - prices[b] + dp[i-1][b])
                #option 2. O(n^2)
                dp[i][j]=max(dp[i][j-1], prices[j]+max_diff) 
                max_diff = max(max_diff, -prices[j]+dp[i-1][j-1])
        #print(dp)
        return dp[k][n-1]

#prices =[2,4,1]
        #[0,0,0]
        #[0,2,2]
        #[0,2,2]