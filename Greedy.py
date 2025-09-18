# In Jump I, you don't force a jump.

# You just need to know:
# 👉 Can I move forward without getting stuck?

# So you still track a "farthest reachable" position (farthest),
# but you don’t need to count jumps or divide the array into "windows."

#  As long as you can greedily move as far as possible at each step,
# ✅ you never need to look back.
# ✅ If you ever reach an index that you cannot reach (i > farthest), you fail immediately.


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        far=nums[0]
        for i in range(len(nums)):
            if i<=far:
                far=max(far, i+nums[i])
            else:
                return False
        return True
    
# At any moment:

# cur_end defines the end of the current jump window.

# You can move freely inside the window (i <= cur_end) without making a new jump.

# You only jump when you reach the end of the window (i == cur_end).

# When you jump, you expand the window to cur_farthest — the farthest place reachable from anywhere in the current window.

# You minimize jumps by greedily walking as far as you can in the current window.

# When window ends, jump once, expand the window — repeat.

def jump(nums):
    jumps = 0         # Number of jumps made
    cur_end = 0       # End of the current jump range
    cur_farthest = 0  # Farthest index reachable

    for i in range(len(nums) - 1):  # IMPORTANT: not including last index
        cur_farthest = max(cur_farthest, i + nums[i])

        # If we have reached the end of the current jump,
        # we need to make another jump
        if i == cur_end:
            jumps += 1
            cur_end = cur_farthest

    return jumps

#garantee uniqueness -> greedy
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas)<sum(cost): #if gas sum is smaller than cost sum, there is no way to complete it
            return -1
        else:#else there must be a starting point to complete it
            remainGas = 0
            start = 0
            for i in range(len(gas)): # starting from the first point
                remainGas+=gas[i]-cost[i]
                if remainGas<0: # if the remainGas ever drop below 0, all starting points before are impossible (only one start is possible, so we just need to find one and stop)
                    start=i+1 #current i is not the start point
                    remainGas=0 #reset remain gas
            return start #since there must be only one solution, the first starting point without getting remainGas<0 until the end of array must be the answer, because all the start before have been excluded, and all the start after will not be the answer because 1. they can be reached by the current start 2. even if they can reach the end as well, only the earlier start will be the answer because a. there is only one answer b. the ealier start must accumulate more remain gas than the later ones, so the earlier one has more gas remaining -> better chance of being the answer!



#O(3n) solution with O(n) space, two passes, one extra list
class Solution:
    def candy(self, ratings: List[int]) -> int:
        res=[1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1] and res[i-1] + 1 > res[i]:
                res[i] = res[i-1] + 1

        for j in range(len(ratings)-2, -1, -1):
            if ratings[j] > ratings[j+1] and res[j+1]+1 > res[j]:
                res[j] = res[j+1] + 1

        return sum(res)