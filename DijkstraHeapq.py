from collections import defaultdict
import heapq

def dijkstraOld(G, startingNode):
	visited = set()
	parentsMap = {}
	pq = []
	nodeCosts = defaultdict(lambda: float('inf'))
	nodeCosts[startingNode] = 0
	heap.heappush(pq, (0, startingNode)) #newCost,Node
 
	while pq:
		# go greedily by always extending the shorter cost nodes first
		_, node = heap.heappop(pq)
		visited.add(node)
 
		for adjNode, weight in G[node].items():
			if adjNode not in visited:
				newCost = nodeCosts[node] + weight
				if nodeCosts[adjNode] > newCost:
						parentsMap[adjNode] = node
						nodeCosts[adjNode] = newCost
						heap.heappush(pq, (newCost, adjNode))
                
	return nodeCosts

def dijkstra(n, edges, src):
    """
    Find shortest paths from source to all vertices using Dijkstra's Algorithm
    Time Complexity: O((V + E) log V) where V is number of vertices, E is number of edges
    Space Complexity: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        #graph[v].append((u, w))  # Add reverse edge for undirected graph
    
    # Initialize distances
    distances = [float('inf')] * n
    distances[src] = 0
    
    # Priority queue to store (distance, vertex)
    pq = [(0, src)]  # (distance, vertex)
    
    while pq:
        curr_dist, curr_vertex = heapq.heappop(pq)
        
        # If we've found a longer path, skip
        if curr_dist > distances[curr_vertex]:
            continue
        
        # Explore neighbors
        for neighbor, weight in graph[curr_vertex]:
            distance = curr_dist + weight
            
            # If we found a shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Test cases
def test_dijkstra():
    # Test case 1
    n1 = 4
    edges1 = [[0,1,4], [0,2,1], [2,1,2], [1,3,1], [2,3,5]]
    src1 = 0
    result1 = dijkstra(n1, edges1, src1)
    print(result1)  # [0, 3, 1, 4]
    assert result1 == [0, 3, 1, 4], "Test case 1 failed"
    
    # Test case 2
    n2 = 2
    edges2 = [[0,1,10]]
    src2 = 0
    result2 = dijkstra(n2, edges2, src2)
    print(result2)  # [0, 10]
    assert result2 == [0, 10], "Test case 2 failed"
    
    # Test case 3
    n3 = 3
    edges3 = [[0,1,1], [1,2,1]]
    src3 = 2
    result3 = dijkstra(n3, edges3, src3)
    print(result3)  # [inf, inf, 0]
    assert result3 == [float('inf'), float('inf'), 0], "Test case 3 failed"
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_dijkstra()



class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import heapq
        min_heap=[]
        for i in nums:
            heapq.heappush(min_heap, i) #put everything in the heap list
            if len(min_heap)>k: 
                # if any number smaller than kth largest num, pop and keep the larger nums in the heap
                heapq.heappop(min_heap)
        return min_heap[0] # return the heap list top


# single minheap with 3 items, push item based on previous popped item
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        from heapq import heappush, heappop
        heap = [[nums1[0]+nums2[0], 0, 0]]
        res = []
        visited = set((0,0)) #you may visit the same index pair twice from two different directions, so keep track the repeated index pair
        n,m = len(nums1), len(nums2)
        for i in range(k):
            s, x, y = heappop(heap) #you cannot pop a value, you need to pop a pair sum
            res.append([nums1[x], nums2[y]])
            #visited.add((x,y)) #too late, you still allow the same index pair to be pushed mutliple times
            if x+1<n and (x+1,y) not in visited:
                heappush(heap, [nums1[x+1]+nums2[y],x+1,y])
                visited.add((x+1,y))
            if y+1<m and (x,y+1) not in visited:
                heappush(heap, [nums1[x]+nums2[y+1],x,y+1]) #push the next only two candiates
                visited.add((x,y+1))
        return res
        

#multiple lists, triple with additional counter in case the val in heap are the same
import heapq
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        res = ListNode()
        curr = res
        smallestHead = []

        counter = 0

        for node in lists:
            if node:
                heapq.heappush(smallestHead, (node.val, counter, node))
                counter += 1

        while smallestHead:
            _, _, node = heapq.heappop(smallestHead)
            if node.next:
                heapq.heappush(smallestHead, (node.next.val, counter, node.next))
                counter += 1
            curr.next = node
            curr=curr.next
            
        return res.next


# two heaps to keep track of the median
import heapq
class MedianFinder:
    def __init__(self):
        self.maxH = []
        self.minH = []

    def addNum(self, num: int) -> None:
        if self.minH == []:
            heapq.heappush(self.minH, num)
        elif self.maxH == []:
            if num > self.minH[0]:
                heapq.heappush(self.minH, num)
            else:
                heapq.heappush(self.maxH, -num)
        else: # if both heap are not empty, check which one to push
            left, right = self.minH[0], -self.maxH[0]
            if num > left: # if num larger than minheap's min, must push left before resizing 
                heapq.heappush(self.minH, num)
            elif num<right:
                heapq.heappush(self.maxH, -num)
            else:
                heapq.heappush(self.minH, num)

        if len(self.minH) - len(self.maxH) > 1:
            m = heapq.heappop(self.minH)
            heapq.heappush(self.maxH, -m)
        elif len(self.maxH) - len(self.minH) > 1:
            m = heapq.heappop(self.maxH)
            heapq.heappush(self.minH, -m)
        

    def findMedian(self) -> float:
        if len(self.maxH) > len(self.minH):
            return -self.maxH[0]
        elif len(self.maxH) < len(self.minH):
            return self.minH[0]
        else:
            return (self.minH[0] - self.maxH[0]) / 2
        
# IPO: Max/Min Two Heap
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        import heapq
        maxProfit = [] #max heap for getting the maximum profits available
        minCap = [(c,p) for c,p in zip(capital, profits)] #min heap for getting all porjects within budget
        heapq.heapify(minCap)

        for _ in range(k):
            while minCap and minCap[0][0] <= w:
                c,p = heapq.heappop(minCap) #pop all the projects within budget since w won't decrease
                heapq.heappush(maxProfit, -p) #push the projects to maxheap to select the most profitable project

            if not maxProfit:# if none of the project are within budget, return earlier
                return w
            project = heapq.heappop(maxProfit)#pop the most profitable project within budget
            w -= project

        return w



# practice for minheap, sorted, zip, gready
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        pairs = [(n1, n2) for n1, n2 in zip(nums1, nums2)]
        pairs = sorted(pairs, key = lambda p: p[1], reverse = True)
        minheap = []
        res = 0
        n1Sum = 0

        for n1, n2 in pairs:
            n1Sum+=n1
            heapq.heappush(minheap, n1) #store new n1 value in the heap

            if len(minheap) > k: #if heap is longer than enough, pop and update answer
                n1Sum-=heapq.heappop(minheap) #pop the smallest number in the heap and subtract from the current sum
                res = max(res, n2*(n1Sum))
            elif len(minheap) == k:#if heap reach k, update result
                res = max(res, n2*(n1Sum))
        return res