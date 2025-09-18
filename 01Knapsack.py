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
        for w in range(capacity + 1):
            # Don't include item i-1
            dp[i][w] = dp[i-1][w]
            # Include item i-1 if possible
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]

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

#-------------------------------------------------------------------------------

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
        # Iterate backwards to avoid reusing one item more than once, just a trick
        for w in range(capacity, weights[i] - 1, -1):
            # dp[w]: the maximum value we can get with w if we use items up to i
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]) 
            
    return dp[capacity]

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