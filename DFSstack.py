graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited = set() # Set to keep track of visited nodes of graph.

def dfsr(visited, graph, node):  # easier
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            # could also use: if neighbour not in visited:
            dfsr(visited, graph, neighbour)


def dfs(visited, graph, node): #avoid recursion-related stack overflow
    stack = [node] 
    visited = set([node]) 
    while stack:
        node = stack.pop()
        print("visiting node", node)
        for i in graph[node]:
            if i not in visited:
                visited.add(i)
                stack.append(i)

# detect number of connected graph using dfs (very classic)
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        visited = set()
        total = 0
        def dfs(node):
            nonlocal total
            if node not in visited:
                visited.add(node)
                for i in range(len(isConnected[node])):
                    if isConnected[node][i] == 1 and i!=node:
                        dfs(i)


        for i in range(len(isConnected)):
            if i not in visited:
                total+=1 #group+1 if find a new start node after doing dfs once
                dfs(i)
        return total


  #DFS with stack + In order traversal without recursion
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left # go to the bottom left and append the path

    def next(self) -> int:
        node = self.stack.pop() # return the bottom left
        curr=node.right #everytime you process a node, add its right node to stack just like inorder(node.left); print(node.val); inorder(node.right)
        while curr:
            self.stack.append(curr)
            curr = curr.left #go to the bottom left of the right node of the bottom left node and start the proper in order traversal

        return node.val
        

    def hasNext(self) -> bool:
        return self.stack!=[]
        

# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()

def create_deep_graph(depth):
    graph = {}
    for i in range(depth):
        graph[str(i)] = [str(i + 1)] if i < depth - 1 else []
    return graph

depth = 2000
deep_graph = create_deep_graph(depth)

# Driver Code
print("Following is the Depth-First Search")
dfsr(visited, deep_graph, '5') #does not work
dfs(visited, deep_graph, '5')



class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n,m = len(grid), len(grid[0])
        islands = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == "1":
                    self.BFS(grid, i, j)
                    islands += 1
        return islands

    def DFS(self, grid, i, j):
        n,m = len(grid), len(grid[0])
        if i>=n or j>=m or i<0 or j<0 or grid[i][j] != "1": #python allow negative indexing!!! so i<0 or j<0 must be added!
            return
        grid[i][j] = "0"  # mark as visited
        self.DFS(grid, i+1, j)
        self.DFS(grid, i, j+1)
        self.DFS(grid, i-1, j)
        self.DFS(grid, i, j-1)

    def BFS(self, grid, i, j):
        n, m = len(grid), len(grid[0])
        queue = deque()
        queue.append((i, j))
        grid[i][j] = "0"  # mark as visited immediately

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # Up, Down, Left, Right
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == "1": #if it is not visited yet
                    queue.append((nx, ny))
                    grid[nx][ny] = "0"  # mark as TOBE visited immediately instead of later when visiting the node

from collections import deque
class Solution:
    def floodFillR(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        # You want to "paint with color 1", but it's already 1.
        # Without early return:
        # DFS at (1,1) calls DFS at (2,1), (0,1), (1,2), (1,0)
        # Then (2,1) calls (3,1), (1,1), (2,2), (2,0)
        # (1,1) again calls (2,1), (0,1), (1,2), (1,0) ... etc
        # It loops forever.
        if image[sr][sc] == color:
            return image

        def dfs(image, i,j, color, initialC):
            n,m = len(image), len(image[0])
            if 0<=i<n and 0<=j<m and image[i][j]==initialC:
                image[i][j]=color
                l = [[0,1],[1,0],[-1,0], [0,-1]]
                for a,b in l:
                    dfs(image, a+i,b+j, color,initialC)

        dfs(image, sr,sc, color, image[sr][sc])
        return image

    def floodFillS(self, image, sr, sc, newColor):
        originalColor = image[sr][sc]
        if originalColor == newColor:
            return image

        rows, cols = len(image), len(image[0])
        stack = [(sr, sc)]

        while stack:
            r, c = stack.pop()
            
            # Boundary check + color match check
            if 0 <= r < rows and 0 <= c < cols and image[r][c] == originalColor:
                image[r][c] = newColor

                # Push neighbors
                stack.append((r+1, c))
                stack.append((r-1, c))
                stack.append((r, c+1))
                stack.append((r, c-1))

        return image


    def floodFillB(self, image, sr, sc, newColor):
        originalColor = image[sr][sc]
        if originalColor == newColor:
            return image

        rows, cols = len(image), len(image[0])
        queue = deque([(sr, sc)])

        while queue:
            r, c = queue.popleft()

            if 0 <= r < rows and 0 <= c < cols and image[r][c] == originalColor:
                image[r][c] = newColor
                queue.append((r+1, c))
                queue.append((r-1, c))
                queue.append((r, c+1))
                queue.append((r, c-1))

        return image

#in place flood fill modification  without visited set + DFS only from border 
class Solution:
    
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == 0:
                return
            grid[r][c] = 0  # mark as water (visited)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                dfs(r + dr, c + dc)
        
        # Step 1: remove all lands connected to border
        for i in range(m):
            for j in range(n):
                if (i == 0 or j == 0 or i == m-1 or j == n-1) and grid[i][j] == 1:
                    dfs(i, j)
        
        # Step 2: count remaining land cells (enclaves)
        return sum(grid[i][j] for i in range(m) for j in range(n))

    def numEnclavesBruteForce(self, grid: List[List[int]]) -> int:
        visited = set()
        total = 0
        def dfs(row, col):
            nonlocal walkoff, cells
            if (row,col) not in visited:
                visited.add((row,col))
                cells+=1
                if row == 0 or row == len(grid)-1 or col==0 or col==len(grid[0])-1:
                    walkoff = True
                for i,j in [[-1,0], [0,-1],[0,1],[1,0]]:
                    if 0<=row+i <=len(grid)-1 and 0<=col+j<=len(grid[0])-1:
                        if grid[row+i][col+j] ==1:
                            dfs(row+i, col+j)

        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i,j) not in visited and grid[i][j] == 1:
                    walkoff = False
                    cells = 0
                    dfs(i,j)
                    if walkoff==False:
                        #print(cells)
                        total += cells
        return total


#use dfs and set as hash, put tuple(list) and tuple in set
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        def dfs(row, col, direction):
            if 0<=row<len(grid) and 0<=col<len(grid[0]) and grid[row][col]==1 and (row,col) not in seen:
                seen.add((row, col))
                currentIsland.append(direction)
                dfs(row, col+1, "r")
                dfs(row+1, col, "d")
                dfs(row, col-1, "l")
                dfs(row-1, col, "u")
                currentIsland.append("b")
                
        seen = set() #keep track all visited coordinates
        uniqueIslands = set() # use hash set to keep track of all unique islands
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                currentIsland = []
                dfs(r,c,"0")
                if currentIsland!= []:
                    uniqueIslands.add(tuple(currentIsland)) #frozenset is order insensitive and inmutable -> hashable, tuple is order sensitive and inmutable ->hashable and unique!
        return len(uniqueIslands)


"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
#dfs + hash + clone
from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        d = {} #In Python, all objects are hashable by default unless:You override __eq__ (equality check) But don’t define __hash__
        def dfs(node):
            if node in d:
                return d[node] #you have to use the original node as clue and retreive the newly created node -> dict!
            c= Node(node.val) #copy itself first
            d[node] = c #marked as visited
            for neighbor in node.neighbors: #copy all its neighbours
                c.neighbors.append(dfs(neighbor)) 
            return c

        return dfs(node) if node else None
    

class Solution: #reversed DFS!!!
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        n, m = len(board), len(board[0])

        def dfs(r,c): #do dfs on all boarder cells since they can't be surrounded
            if r<0 or c<0 or r==n or c==m or board[r][c]!="O":
                return
            board[r][c] = "T"
            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        for i in range(n): #check the boarder
            for j in range(m):
                if board[i][j]=="O" and (i in [0,n-1] or j in [0, m-1]):
                    dfs(i,j)

        for i in range(n): #check the whole board, convert surrounded O to X
            for j in range(m):
                if board[i][j]=="O":
                    board[i][j]="X"

        for i in range(n): #check the whole board, convert T back to O
            for j in range(m):
                if board[i][j]=="T":
                    board[i][j]="O"
            
        
        

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = [] #store all the intermediate results and latest numbers
        for token in tokens: # process tokens from the bottom up: left to right
            if token in {"+", "-", "*", "/"}:
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    # Truncate toward zero
                    stack.append(int(a / b))
            else:
                stack.append(int(token))
        return stack[0]
        

class Solution:
    def calculate(self, s: str) -> int:
        sign = 1
        res=0 # accumulate partial res in ()
        stack = [] # handle multi level ()
        num = 0
        for i in s:
            if i.isdigit():
                num = num*10 + int(i) # handle number greater than 9
            elif i == "+":
                res+= sign * num # add the number before +
                num = 0 # reset num for number greater than 9 handle
                sign = 1 # set the sign for the next num
            elif i=="-":
                res+= sign * num # add the number before -
                num = 0 # reset num for number greater than 9 handle
                sign = -1 # set the sign for the next num
            elif i=="(":
                stack.append(res) # store the previous res/sign in stack before reset res
                stack.append(sign)
                sign = 1 # reset sign for new ()
                res=0 # reset res because you need a fresh accumulator
            elif i==")":
                res+= sign * num # early compute for the last num in ()
                num = 0 # reset num for number greater than 9 handle

                res *= stack.pop() #correct sign of res inside ()
                res += stack.pop() #reset res to be the sum of the num inside and outside the ()
            else: # ignore weird spaces
                continue
        return res+ sign * num # handle the last num and one num case
        #1+(4+5+2)-3

class Solution:
    def decodeString(self, s: str) -> str:

        stack = []

        for c in s:
            if c!="]":
                stack.append(c) # add chars to the stack first
            else:
                substr = ""
                while stack[-1] != "[": # Finding the current substr by popping
                    substr = stack.pop() + substr
                stack.pop() #pop "["

                n = ""
                while stack and stack[-1].isdigit():
                    n = stack.pop() + n

                stack.append(int(n) * substr) #replace the substr with the new expanded one

        return "".join(stack)
        
#monotonically decreasing stack with index: whenever pop, it mean a day is sucessfully processed -> a hotter day is found
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures) #intialize the res with 0s
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t>stack[-1][1]: #pop and update the res when current temperature is greater than the last unprocessed day (didn't find a hotter day before) because a hotter day than some previous days is found 
                idx, temp = stack.pop() #remove from the stack mean a hotter day is found this idx day
                res[idx] = i-idx
            stack.append([i, t])
        return res


#monotonic decreasing stack using answers (each value is used only twice)
class StockSpanner:

    def __init__(self):
        self.stack = []
        

    def next(self, price: int) -> int:
        ans = 1
        while self.stack and self.stack[-1][0] <= price:
            ans += self.stack.pop()[1] #pop all previous values smaller that the current price to keep the stack decreasing
        self.stack.append([price, ans])
        return ans


# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)


# hwo to use yield and yield from
class Solution:
    def leafSimilarDFS(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        res = []
        def dfs(root):
            if root==None:
                return 
            elif root.left==root.right==None:
                res.append(root.val)
            else:
                dfs(root.left)
                dfs(root.right)
        dfs(root1)
        res1=res
        res=[]
        dfs(root2)
        return res==res1
        
    def leafSimilar(self, root1, root2):
        def dfs(node):
            if node:
                if not node.left and not node.right:
                    yield node.val
                # for val in dfs(node.left):
                #     yield val
                # for val in dfs(node.right):
                #     yield val
                yield from dfs(node.left)
                yield from dfs(node.right)

        return list(dfs(root1)) == list(dfs(root2))