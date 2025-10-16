# 3 tricks for 0/1 Knapsack: 
# 1. come up with the 2d solution: include item i-1 or not, then we can use the dp[i-1] row to update the dp[i] row
# 2. optimize it to 1d by overwritting all the rows 
# 3. iterate backwards to use the correct dp values when looking backwards in dp[w - weights[i]]


def knapsack_2d(weights, values, capacity):
    """
    0/1 Knapsack using 2D DP
    Time Complexity: O(n * capacity)
    Space Complexity: O(n * capacity)
    """
    n = len(weights)
    # dp[i][w] = max value using items 0 to i-1 with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1): #no need to loop backward because we are using the dp[i-1] row, which is fixed
            # Don't include item i-1
            dp[i][w] = dp[i-1][w]
            # Include item i-1 if possible
            if w >= weights[i-1]: 
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]



def knapsack_1d(weights, values, capacity): #we only care the case if we can consider all items in the bag
    """
    0/1 Knapsack using 1D DP
    Time Complexity: O(n * capacity)
    Space Complexity: O(capacity)
    """
    n = len(weights)
    # dp[w] = max value achievable with capacity w
    dp = [0] * (capacity + 1)
    
    # Process each item
    for i in range(n):
        # Iterate backwards to avoid reusing the ith item more than once: dp[w - weights[i]] must be the max value achievable with capacity [w - weights[i]] without using the ith item
        for w in range(capacity, weights[i] - 1, -1):
            # dp[w]: the maximum value we can get with w if we use items up to i
            # for every ith iteration, we update all dp[w] with the maximum value we can get with w if we use items up to i
            # dp[w] is either the dp[w] (does not include current item i) or the dp[w - weights[i]] + values[i] (includes current item i)

            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]) 
            
    return dp[capacity]

#-------------------------------------------------------------------------------

# Test cases
def test_knapsack_2d():
    # Test case 1
    weights1 = [1, 2, 3]
    values1 = [6, 10, 12]
    capacity1 = 5
    assert knapsack_2d(weights1, values1, capacity1) == 22, "Test case 1 failed"
    
    # Test case 2
    weights2 = [1, 3, 4, 5]
    values2 = [1, 4, 5, 7]
    capacity2 = 7
    assert knapsack_2d(weights2, values2, capacity2) == 9, "Test case 2 failed"
    
    print("All 2D test cases passed!")


# Test cases
def test_knapsack_1d():
    # Test case 1
    weights1 = [1, 2, 3]
    values1 = [6, 10, 12]
    capacity1 = 5
    assert knapsack_1d(weights1, values1, capacity1) == 22, "Test case 1 failed"
    
    # Test case 2
    weights2 = [1, 3, 4, 5]
    values2 = [1, 4, 5, 7]
    capacity2 = 7
    assert knapsack_1d(weights2, values2, capacity2) == 9, "Test case 2 failed"
    
    print("All 1D test cases passed!")

if __name__ == "__main__":
    test_knapsack_1d()
    test_knapsack_2d()



# 1d 0/1 knapsack with true/false
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total%2 == 1:
            return False
        target = total // 2 
        dp = [False] * (target + 1)
        dp[0] = True # always possible to reach 0 target
        for i in nums:
            for j in range(target, i - 1, -1):
                dp[j] = dp[j] | dp[j-i] #either reaching the target without current i element or reaching the target after adding current i
        return dp[-1]

# Unbounded 0/1 knapsack without need to loop backward because any item can be used infinite times and thus can be double counted
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


# Unbounded 0/1 knapsack without need to loop backward because any item can be used infinite times and thus can be double counted
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount+1) # number of way to reach this amount by using coins upto i (don't have to use i)
        dp[0] = 1
        for i in coins:
            for n in range(i, len(dp)):          
                dp[n]+=dp[n-i] #the new ways include all the original ways (without using i) and new number of ways after using i (dp[n-i])
        return dp[-1]

# 0/1 Knapsack with random unordered update in row: 2D->1D 
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [float('-inf')] * 3
        dp[0] = 0  # Base case: empty sum is 0
        
        for num in nums:
            new_dp = dp[:]            
            for r in range(3):
                # Take current number, calculate the new remainder after adding r and update dp at the new remainder position
                new_dp[(r + num) % 3] = max(new_dp[(r + num) % 3], dp[r] + num)
            
            dp = new_dp
        
        return dp[0] if dp[0] != float('-inf') else 0

    def maxSumDivThree2(self, nums: List[int]) -> int:
        n = len(nums)
        
        # dp[i][r] = maximum sum using first i elements with sum % 3 == r
        dp = [[float('-inf')] * 3 for _ in range(n + 1)]
        
        # Base case: using 0 elements, sum = 0 (remainder 0)
        dp[0][0] = 0
        
        for i in range(1, n + 1):         
            for r in range(3):
                # if nums[i-1]%3 == 0:
                #     dp[i][r] = dp[i-1][r] + nums[i-1]
                # else:

                #(nums[i-1] % 3 + x) %3 = r -> x = (r - nums[i-1]) % 3  
                complementary_remainder = (r - nums[i-1]) % 3 
                
                #if dp[i-1][prev_remainder] != float('-inf'):
                # Option 1: Skip current element                  
                # Option 2: Take current element
                dp[i][r] = max(dp[i-1][r], dp[i-1][complementary_remainder] + nums[i-1])
        
        return dp[n][0] if dp[n][0] != float('-inf') else 0