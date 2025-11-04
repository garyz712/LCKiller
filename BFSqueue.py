



from collections import deque
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

def bfs(graph, node):
  toBeVisited = set([node]) # List to keep track of visited nodes.
  queue = deque([node])     #Initialize a queue
  while queue:
    node = queue.popleft()
    print("visiting", node) #officially processing a node
    for i in graph[node]:
      if i not in toBeVisited:
        queue.append(i)
        toBeVisited.add(i) # add it to the To Do List, but check that it is seen so that it will not be added to queue again

        
if __name__ == "__main__":
  # Driver Code
  bfs(graph, 'A')


def bfsSuboptimal(graph, startUser, endUser): # May enqueue same node twice
  queue = deque([startUser])
  visited = set()

  while not queue.isEmpty():
      user = queue.dequeue()
      
      if user == endUser:
          return True

      if user in visited:
          continue

      visited.add(user)

      for neighbor in graph[user]:
          if neighbor not in visited:
              queue.enqueue(neighbor)

  return False




  #layer by layer processing, each layer run two for loops to get all the childrens
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)

        if endWord not in wordSet:
            return 0

        queue = deque([beginWord])
        level=1

        l = len(beginWord)

        while queue:
            for i in range(len(queue)): #make sure to process all words in this level
                word = queue.popleft()

                if word == endWord:
                    return level

                #for every character in this word, process all characters
                for j in range(l):
                    for c in "abcdefghijklmnopqrstuvwxyz":
                        newword = word[:j]+c+word[j+1:]
                        
                        if newword in wordSet:
                            queue.append(newword)
                            wordSet.remove(newword) #remove the word from the set to avoid duplicate processing
                
            level+=1 # proceed to the next layer

        return 0

#layer by layer processing, each layer run two for loops to get all the childrens
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank = set(bank)
        #visited = set()
        queue = deque([startGene])
        layer=0

        while queue:
            length = len(queue)

            for i in range(length):
                current = queue.popleft() #not pop()!
                #visited.add(current)

                if current == endGene:
                    return layer

                for j in range(8):
                    for c in "ACGT":
                        newGene = current[:j] + c + current[j+1:]

                        if newGene in bank:                           
                            #if newGene not in visited: no need for this since bank can filter all the visited genes
                                queue.append(newGene)
                                bank.remove(newGene)

            layer+=1

        return -1

#layer wise BFS with tuple and matrix
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        length = len(board)

        def intToPos(id_):
            id0 = id_ - 1
            r = length - 1 - id0 // length
            c = id0 % length
            # if row index from bottom is odd, go right->left, length might be odd or even
            if ((length - 1 - r) % 2) == 1:
                c = length - 1 - c
            return r, c

        from collections import deque
        queue = deque([(1, 0)])
        visited = {1}
        
        while queue:
            number, dices = queue.popleft()
            if number == length**2:
                return dices
            else:
                for i in range(1,7):
                    if number+i>length**2:
                        break
                    r,c = intToPos(number+i)
                    if board[r][c]!=-1 and board[r][c] not in visited:
                        visited.add(board[r][c])
                        queue.append((board[r][c], dices+1))
                    elif board[r][c]==-1 and number +i not in visited:
                        visited.add(number +i)
                        queue.append((number +i, dices+1))
        return -1
                 
                 

#weighted BFS
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        from collections import defaultdict
        #build the bidirected graph first
        graph = defaultdict(list)
        for i, eq in enumerate(equations):
            nom, denom = eq
            graph[nom].append([denom, values[i]])
            graph[denom].append([nom, 1/values[i]])

        def bfs(src, target):
            if src not in graph or target not in graph:
                return -1
            queue = deque([[src, 1]]) #src->src: 1
            visited = set([src])

            while queue:
                denom, quotient = queue.popleft()
                if denom == target:
                    return quotient

                for pair in graph[denom]:
                    node, val = pair
                    if node not in visited: #don't process nodes already in TODO queue
                        queue.append([node, quotient*val]) #next node, current quotient
                        visited.add(denom)
            return -1 #if there is no such path from src->target
        return [bfs(q[0], q[1]) for q in queries]





# iterate FIFO queue processing with 4 counters or two queue
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        queue = deque(senate)
        # for c in senate:
        #     queue.append(c)
        Daccum = 0
        Raccum = 0
        # Keep counts of active senators to check termination efficiently
        Dcount = senate.count("D")
        Rcount = senate.count("R")

        while Dcount > 0 and Rcount > 0:
            firstParty = queue.popleft()
            if firstParty == "D":

                if Raccum==0:
                    queue.append(firstParty)
                    Daccum += 1
                else:
                    Raccum-=1
                    Dcount-=1
            else:
                if Daccum==0:
                    queue.append(firstParty)
                    Raccum += 1 
                else:
                    Daccum -= 1
                    Rcount -= 1
        return "Radiant" if queue[0] == "R" else "Dire"



# when to add node to toBeVisited list make a difference!
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        queue = collections.deque([tuple(entrance)])
        count = 0
        foundexit = False
        while queue:
            n = len(queue)
            for i in range(n):
                cell = queue.popleft()
                # don't mark it as visited here!
                #and  → higher precedence than →  or So A or B and C is evaluated as A or (B and C)
                if (cell[0] == 0 or cell[0] == len(maze)-1 or cell[1] == 0 or cell[1] == len(maze[0])-1) and cell!=tuple(entrance):
                    foundexit = True
                    return count

                for i, j in [[1,0],[0,1],[-1,0],[0,-1]]:
                    if 0<=cell[0]+i<=len(maze)-1 and 0<=cell[1]+j<=len(maze[0])-1 and  maze[cell[0]+i][cell[1]+j]==".":
                        maze[cell[0]+i][cell[1]+j] = "+" #must add it to (TO BE) visited set here to save time from double adding instead of where actually visiting the cell
                        queue.append((cell[0]+i, cell[1]+j))

            count+=1
        return count if foundexit else -1


# mixed BFS + DFS
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        
        m, n = len(maze), len(maze[0])
        visited = set()
        q = [(start[0], start[1])] # only store stop points
        visited.add((start[0], start[1]))
        
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        while q:
            x, y = q.pop(0)  # BFS
            
            if [x, y] == destination:
                return True
                
            for dx, dy in dirs:
                nx, ny = x, y
                # Roll in this direction until hit wall or boundary
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0: # do partial DFS to reach a stop point
                    nx += dx
                    ny += dy
                # Now (nx, ny) is the stopping point
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
                # if the stop point is visited, don't add it to the queue
        
        return False