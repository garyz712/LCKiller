from collections import defaultdict, deque
 
#Find a valid topological order with cycle detection! BFS recommended!
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        order = []
        graph = [[] for i in range(numCourses)]
        indeg = [0] * numCourses
        for a,b in prerequisites:
            graph[b].append(a)
            indeg[a]+=1

        queue=deque([i for i in range(numCourses) if indeg[i]==0])
        count=0
        while queue:
            course = queue.popleft()
            count+=1
            order.append(course)

            for n in graph[course]:
                indeg[n]-=1
                if indeg[n]==0:
                    queue.append(n)

        return [] if count!=numCourses else order
    
#Class to represent a graph
class Graph:
    def __init__(self,vertices):
        self.graph = defaultdict(list) #dictionary containing adjacency List
        self.V = vertices #No. of vertices
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # A recursive DFS function used by topologicalSort
    def topologicalSortUtil(self,v,visited,stack): 
 
        # Mark the current node as visited.
        visited[v] = True
 
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # Push current vertex to stack which stores result
        stack.insert(0,v)
 
    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack =[]
 
        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # Print contents of stack
        print (stack)
 
g= Graph(6)
g.addEdge(5, 2);
g.addEdge(5, 0);
g.addEdge(4, 0);
g.addEdge(4, 1);
g.addEdge(2, 3);
g.addEdge(3, 1);
 
print ("Following is a Topological Sort of the given graph")
g.topologicalSort()

    

#Use topological sort to detect a cycle in Directed Graph
#Prefer BFS Kahn's Algorithm by default for topological sort interview questions unless DFS is specifically easier.
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Kahn's Algorithm: Build the graph and indegree array as before
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses #track len([b,c,d,etc]) -> a

        for a, b in prerequisites:
            graph[b].append(a)  # Edge from b → [a,c,d,etc], notice do not use graph[b] = a !
            indegree[a] += 1

        # Find all courses that have no prerequisites (indegree == 0) — these can be taken immediately. 
        queue = deque([i for i in range(numCourses) if indegree[i] == 0])

        count = 0 # Keep track of how many courses we have successfully taken.
        
        # For courses with no prerequisites left at the moment.
        while queue:
            #Pop a course from the queue (take the course).
            course = queue.popleft()
            #Increment count (we completed one course).
            count += 1  

            #For each neighbor (i.e., courses that depend on the current course):
            for neighbor in graph[course]:
                # Decrease their indegree (because now they have one less prerequisite).
                indegree[neighbor] -= 1  

                #If a neighbor's indegree becomes 0, 
                #that means it is ready to be taken → push it into the queue.
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        # If we finished all courses, return True
        # If some courses are stuck with unmet prerequisites → there's a cycle → return False.
        return count == numCourses

    def canFinishDFS(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        #many courses may share the same prereq, a course can have multiple prereq
        for a, b in prerequisites:
            graph[b].append(a) #idx b is prereq, a is course: b->a 
        visited = [0] * numCourses #no need to pass visited to dfs as reference

        def dfs(node): #No need to use (self,...) use dfs to find a cycle 
            #if you revisit a node that is still being visited (state 1),
            #it means you have found a cycle in the graph!
            if visited[node] == 1:
                return False  # found cycle

            #You already finished exploring everything downstream from this node,
            #and it’s safe (no cycles in its descendants).
            if visited[node] == 2:
                return True  # already processed
            
            # if visited[node] == 0, start processing it
            visited[node] = 1  # mark as visiting

            #For all the neighbors (outgoing edges) from this node, 
            #if any neighbor’s DFS finds a cycle, immediately return False to indicate a cycle exists.
            for neighbor in graph[node]: 
                if not dfs(neighbor):
                    return False
            visited[node] = 2  # If there is no cycle starting from node, mark it as safe and return True
            return True

        # The graph might be disconnected! There might be multiple independent groups of courses (subgraphs), and you need to check all of them!
        for i in range(numCourses):
            if not dfs(i): #If this node is in a cycle, return False
                return False
        return True #If all nodes are not in cycles, prereq can be satisfied



class Solution:
    def alienOrder(self, words: List[str]) -> str:
        from collections import defaultdict, deque
        graph = defaultdict(set)

        # Collect all unique characters
        chars = set()
        for word in words:
            for c in word:
                chars.add(c)

        indegree = {c:0 for c in chars}#don't use defaultdict(int) as it will miss some elements in queue
        #print(indegree)
         
        #build the dependency graph
        for i in range(1, len(words)):
            shorter = min(len(words[i]), len(words[i-1]))
            if len(words[i-1]) > len(words[i]) and words[i-1][: len(words[i])] == words[i]: #check the edge case where the prefix match
                return ""
            for j in range(shorter):
                if words[i][j] != words[i-1][j]: 
                    if words[i][j] not in graph[words[i-1][j]]: #avoid add duplicates to indegree
                        graph[words[i-1][j]].add(words[i][j]) #prev->next 
                        indegree[words[i][j]]+=1
                
                    break #must match the outer if statement instead of the inner one

        #print(indegree)
        queue = deque([key for key in chars if indegree[key]==0])
        #print(graph)
        order = ""
        while queue:
            char = queue.popleft()
            order+=char
            for nextNeigh in graph[char]:
                indegree[nextNeigh] -= 1

                if indegree[nextNeigh] == 0:
                    queue.append(nextNeigh)
        

        return "" if len(order) != len(chars) else order

# post order DFS and return with reversed order 
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # Step 1: Collect all unique characters
        chars = set()
        for word in words:
            for c in word:
                chars.add(c)
        
        # Step 2: Build the dependency graph
        graph = defaultdict(set)
        for i in range(1, len(words)):
            word1, word2 = words[i-1], words[i]
            if len(word1) > len(word2) and word1[:len(word2)] == word2:
                return ""
            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    graph[word1[j]].add(word2[j])
                    break
        
        # Step 3: DFS with cycle detection
        visited = {}  # None: unvisited, 1: in recursion stack, 2: fully processed
        order = []
        
        def dfs(char):
            if char in visited:
                if visited[char] == 1:  # Cycle detected
                    return False
                return True  # Already processed
            visited[char] = 1  # Mark as in recursion stack
            for neighbor in graph[char]:
                if not dfs(neighbor):  # Cycle in neighbor
                    return False
            visited[char] = 2  # Mark as fully processed
            order.append(char)  # Post-order: add after processing neighbors
            return True
        
        # Step 4: Run DFS on all characters
        for c in chars:
            if c not in visited:
                if not dfs(c):
                    return ""
        
        # Step 5: Reverse the post-order result
        return "".join(order[::-1]) if len(order) == len(chars) else ""


#Check postorder uniqueness
from collections import defaultdict, deque

class Solution:
    def sequenceReconstruction1(self, nums: List[int], sequences: List[List[int]]) -> bool:
        numset = set()
        graph = defaultdict(list)
        for seq in sequences:
            numset.update(seq)
            for i in range(len(seq)-1):
                graph[seq[i]].append(seq[i+1]) 
        order = []
        visited = {}
        #print(graph)

        # Check Uniqueness: if there is an edge that does not exist in graph, that means the nums order can flip around this edge, the order is not unique
        for i in range(len(nums)-1):
            if nums[i+1] not in graph[nums[i]]:
                return False

        def dfs(num):
            if num in visited:
                if visited[num]==1:
                    #print("0")
                    return False #there is a cycle detected
                else:
                    return True #there is no cycle until that node is processed
            else:
                visited[num] = 1
                for neighbor in graph[num]:
                    if not dfs(neighbor):
                        return False
                visited[num] = 2 
                order.append(num)
                return True

        return dfs(nums[0]) and order[::-1] == nums

    def sequenceReconstruction(self, org, seqs):
        graph = defaultdict(set)
        indegree = defaultdict(int)
        
        # Initialize nodes
        for seq in seqs:
            for num in seq:
                if num not in indegree:
                    indegree[num] = 0
        
        # Build graph
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                if b not in graph[a]:
                    graph[a].add(b)
                    indegree[b] += 1
        
        # If numbers mismatch
        if len(indegree) != len(org):
            return False
        
        # BFS
        queue = deque([node for node in indegree if indegree[node] == 0])
        result = []
        
        while queue:
            if len(queue) > 1:
                return False  # Multiple orders possible
            node = queue.popleft()
            result.append(node)
            for nei in graph[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    queue.append(nei)
        
        return result == org

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        inorder = [0] * n
        graph = defaultdict(list) #{} is wrong because some courses does not have next course

        for a,b in relations:
            if a in graph:
                graph[a].append(b)              
            else:
                graph[a] = [b]
            inorder[b-1]+=1

        queue = deque([i+1 for i in range(len(inorder))if inorder[i]==0])
        res=0
        total=0
        while queue:
            for i in range(len(queue)):
                course = queue.popleft()
                total+=1
                for nxt in graph[course]:
                    inorder[nxt-1] -=1
                    if inorder[nxt-1]==0: #this ensure that a course will be added to the queue only once
                        queue.append(nxt)
            res+=1
        #print(total)
        return res if total==n else -1

