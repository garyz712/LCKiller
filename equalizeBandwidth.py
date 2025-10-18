


def equalizeBandwidth(servers):
    target = max(servers) # target could be max(servers)+1 as well if changing the max is allowed
    upgrade2 = 0                    # Count of "+2" upgrade units needed
    upgrade1 = 0                    # Count of "+1" upgrade units needed
    total = 0                    # Total upgrade units needed
    
    for bw in servers:
        diff = target - bw            # Difference to reach max bandwidth
        total += diff               # Add to total difference
        upgrade2 += diff // 2          # Add count of "+2" upgrades needed
        upgrade1 += diff % 2           # Add count of "+1" upgrades needed
    
    if total == 0:               # If all servers are already equal
        return 0
    
    l, r = 0, 2 * total          # Binary search range
    while l < r:
        mid = (l + r) // 2
        if check(mid, upgrade2, upgrade1):
            r = mid
        else:
            l = mid + 1
    
    return l


def check(hours, upgrade2, upgrade1):
    even_hours = hours // 2                # Number of even hours (for +2 upgrades)
    odd_hours = hours - even_hours         # Number of odd hours (for +1 upgrades)
    moreUpgrade2Needed = upgrade2 - even_hours    # Remaining "+2" upgrades after using even hours
    if moreUpgrade2Needed < 0:             # If we have more even hours than needed
        moreUpgrade2Needed = 0             # don't use these even hours since that will exceed the target
    Upgrade1Needed = moreUpgrade2Needed * 2 + upgrade1  # If we don't have enough even hours, convert remaining "+2" upgrades to "+1" + original "+1" upgrades
    return Upgrade1Needed <= odd_hours     # Check if we have enough odd hours


# binary search
import math
def equalizeBandwidth2(servers):
    # Base case: if all servers have the same bandwidth
    if len(set(servers)) == 1:
        return 0
    
    max_bw = max(servers)
    total_diff = sum(max_bw - bw for bw in servers)

    # If we only use +1 upgrades on odd hours
    right = total_diff * 2
    # Binary search for the minimum hours needed
    left = 0

    
    while left < right:
        mid = (left + right) // 2
        if is_possible(servers, mid):
            right = mid
        else:
            left = mid + 1
    
    return left

def is_possible(servers, hours):
    # Calculate how many upgrades of each type we have
    light_upgrades = (hours + 1) // 2  # Odd hours (+1 Mbps)
    heavy_upgrades = hours // 2        # Even hours (+2 Mbps)
    
    # Calculate the maximum bandwidth we can reach
    target = max(servers)
    
    # Calculate how much each server needs to increase
    # to match the target bandwidth
    differences = [target - server for server in servers]
    
    # Sort differences in descending order (greedy approach)
    differences.sort(reverse=True)
    
    # Try to satisfy each server's needs
    for diff in differences:
        if diff == 0:
            continue  # Skip servers already at target
        
        # Use heavy upgrades first (more efficient)
        h_used = min(diff // 2, heavy_upgrades)
        diff -= h_used * 2
        heavy_upgrades -= h_used
        
        # Use light upgrades for remainder
        l_used = min(diff, light_upgrades)
        diff -= l_used
        light_upgrades -= l_used
        
        # If we couldn't fully upgrade this server, return False
        if diff > 0:
            return False
    
    return True


import heapq
from collections import Counter
def equalizeBandwidth1(servers):
    # Step 1: Compute the target bandwidth (maximum value)
    target = max(servers)   
    # Step 2: Compute the initial gaps for each server
    gaps = [target - s for s in servers]   
    # If all gaps are 0, we're already done
    if all(gap == 0 for gap in gaps):
        return 0   
    # Step 3: Simulate the process hour by hour
    hour = 0
    while any(gap > 0 for gap in gaps):  # Continue until all gaps are 0
        hour += 1
        is_odd = hour % 2 == 1  # Odd hour: Light upgrade; Even hour: Heavy upgrade
        found_gap_2 = False
        found_gap_1 = False
        found_big_gap = False
        big_gap_idx = None
    
        if is_odd:  # Odd hour: Light upgrade (+1 Mbps)
            # Search for a gap of 1
            for i in range(len(gaps)):
                if gaps[i] == 1:
                    gaps[i] -= 1  # Apply Light upgrade
                    found_gap_1 = True
                    break
                elif gaps[i] == 2:
                    found_gap_2 = True
                elif gaps[i] > 2:
                    found_big_gap = True
                    big_gap_idx = i           
            # If no gap of 1 was found but there’s a gap of 2, skip this hour
            if not found_gap_1 and found_gap_2 and not found_big_gap:
                print("skipped")
                continue  # Skip the hour (we’ll handle gap=2 on the next even hour)
            elif not found_gap_1 and found_big_gap:
                gaps[big_gap_idx] -= 1  # Apply Light upgrade to reduce gap

        
        else:  # Even hour: Heavy upgrade (+2 Mbps)
            # Search for a gap of 2
            for i in range(len(gaps)):
                if gaps[i] == 2:
                    gaps[i] -= 2  # Apply Heavy upgrade
                    found_gap_2 = True
                    break
                if gaps[i] == 1:
                    found_gap_1 = True
                if gaps[i] > 2:
                    found_big_gap = True
                    big_gap_idx = i
            
            # If no gap of 2 was found but there’s a gap of 1, skip this hour
            if not found_gap_2 and found_gap_1 and not found_big_gap:
                print("skipped")
                continue  # Skip the hour (we’ll handle gap=1 on the next odd hour)
            elif not found_gap_2 and found_big_gap:
                # If no gap of 2 was found, check for gaps ≥ 3
                gaps[big_gap_idx] -= 2  # Apply Heavy upgrade to reduce gap
        print(gaps)
    
    return hour


def getPlaylistCount(videoLengths, n, k, threshold):
    # Step 1: Count elements > threshold in the first window of size k
    count_exceeding = 0
    for i in range(k):
        if videoLengths[i] > threshold:
            count_exceeding += 1
    
    # Step 2: Initialize the result
    playlists = 0
    if count_exceeding == 0:
        playlists += 1  # First window is valid
    
    # Step 3: Slide the window and update the count
    for i in range(k, n):
        # Remove the leftmost element of the previous window
        if videoLengths[i - k] > threshold:
            count_exceeding -= 1
        # Add the new rightmost element to the window
        if videoLengths[i] > threshold:
            count_exceeding += 1
        # If no elements in the current window exceed the threshold, increment playlists
        if count_exceeding == 0:
            playlists += 1
    
    return playlists

def test_getPlaylistCount():
    # Test Case 0: Sample Case 0 from the problem
    videoLengths = [1, 3, 2, 4, 3]
    n = 5
    k = 3
    threshold = 100
    assert getPlaylistCount(videoLengths, n, k, threshold) == 3, "Test Case 0 Failed"
    print("Test Case 0 Passed: Expected 3, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 1: Sample Case 1 from the problem
    videoLengths = [6, 10, 34, 24, 12, 21, 30, 35]
    n = 8
    k = 2
    threshold = 30
    assert getPlaylistCount(videoLengths, n, k, threshold) == 4, "Test Case 1 Failed"
    print("Test Case 1 Passed: Expected 4, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 2: Example from the problem description
    videoLengths = [3, 1, 5, 6, 8, 2, 1]
    n = 7
    k = 2
    threshold = 5
    assert getPlaylistCount(videoLengths, n, k, threshold) == 3, "Test Case 2 Failed"
    print("Test Case 2 Passed: Expected 3, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 3: Edge Case - k equals n (entire array as one window)
    videoLengths = [1, 2, 3]
    n = 3
    k = 3
    threshold = 5
    assert getPlaylistCount(videoLengths, n, k, threshold) == 1, "Test Case 3 Failed"
    print("Test Case 3 Passed: Expected 1, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 4: Edge Case - k equals 1 (single element windows)
    videoLengths = [1, 10, 3, 20, 5]
    n = 5
    k = 1
    threshold = 10
    assert getPlaylistCount(videoLengths, n, k, threshold) == 4, "Test Case 4 Failed"
    print("Test Case 4 Passed: Expected 3, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 5: All elements exceed threshold
    videoLengths = [100, 200, 300, 400]
    n = 4
    k = 2
    threshold = 50
    assert getPlaylistCount(videoLengths, n, k, threshold) == 0, "Test Case 5 Failed"
    print("Test Case 5 Passed: Expected 0, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 6: All elements are below threshold
    videoLengths = [1, 2, 3, 4, 5]
    n = 5
    k = 2
    threshold = 10
    assert getPlaylistCount(videoLengths, n, k, threshold) == 4, "Test Case 6 Failed"
    print("Test Case 6 Passed: Expected 4, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 7: Large threshold, all elements pass
    videoLengths = [1000, 2000, 3000, 4000]
    n = 4
    k = 2
    threshold = 10**9
    assert getPlaylistCount(videoLengths, n, k, threshold) == 3, "Test Case 7 Failed"
    print("Test Case 7 Passed: Expected 3, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 8: Mixed values with small k
    videoLengths = [5, 15, 25, 3, 7, 20]
    n = 6
    k = 3
    threshold = 10
    assert getPlaylistCount(videoLengths, n, k, threshold) == 0, "Test Case 8 Failed"
    print("Test Case 8 Passed: Expected 1, Got", getPlaylistCount(videoLengths, n, k, threshold))

    # Test Case 9: Single element array
    videoLengths = [5]
    n = 1
    k = 1
    threshold = 10
    assert getPlaylistCount(videoLengths, n, k, threshold) == 1, "Test Case 9 Failed"
    print("Test Case 9 Passed: Expected 1, Got", getPlaylistCount(videoLengths, n, k, threshold))

    print("All test cases passed!")

# Run the tests
test_getPlaylistCount()

# Test the case
servers = [2, 2, 2, 3]
print(equalizeBandwidth(servers))  # Should output 5

# Test cases
print(equalizeBandwidth([1, 2, 4]))  # Output: 4
print(equalizeBandwidth([2, 2, 2 , 3]))  # Output: 5
print(equalizeBandwidth([1, 2 , 3]))  # Output: 2
print(equalizeBandwidth([2, 3]))     # Output: 1
print(equalizeBandwidth([2, 4]))     # Output: 2
print(equalizeBandwidth([2, 7]))     # Output: 4 [3,7] -[5,7]- / - [7,7]
print(equalizeBandwidth([1, 2, 6]))  # Output: 6
#[2,2,6] - [4,2,6] - [4,3,6] - [4,5,6] - [4,6,6]- [6, 6, 6]
print(equalizeBandwidth([1, 2, 7]))  # Output: 8
#[2,2,7] [4,2,7] [4,3,7] [4,5,7] [5,5,7] [7,5,7] \ [7,7,7] 
print(equalizeBandwidth([3, 5, 8, 10])) #10
print(equalizeBandwidth([1, 1, 2, 2, 4, 5]))  # Output: 10