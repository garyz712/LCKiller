# matrix transpose + hashmap
# zip(*grid) is the transpose = [list(col) for col in zip(*grid)]
# # equivalent to:
# zip([3,2,1], [1,7,6], [2,7,7])
# OR
# import numpy as np
# grid_np = np.array(grid)
# transpose = grid_np.T.tolist()

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        from collections import Counter
        rows = Counter(tuple(row) for row in grid)
        cols = Counter(tuple(col) for col in zip(*grid))
        res = 0
        for k in rows:
            res += rows[k] * cols.get(k, 0)
        return res
        
# validate if symmetric word matrix
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        n = len(words)
        for i in range(n):
            for j in range(len(words[i])):
                try:
                    if words[i][j] != words[j][i]:
                        return False
                except:
                    return False
        return True

# in place matrix modification
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
    
        """
        n, m = len(matrix), len(matrix[0])
        first_row = 1 #O(1) space to record if the first row is 0
        for i in range(n):
            for j in range(m):
                if matrix[i][j]==0:
                    if i==0:
                        first_row = 0
                        matrix[0][j] = 0
                    else:
                        matrix[0][j] = 0
                        matrix[i][0] = 0

        #must first check and modify 2nd-last row which will not modify the first row which has the zero column record
        for i in range(1,n):
            if matrix[i][0]==0:
                matrix[i] = [0]*m

        #print(matrix)
        for j in range(m):
            if matrix[0][j]==0:
                for i in range(n):
                    matrix[i][j]=0
        
        #print(matrix)
        #must be the last thing to check because once zero, the first row will be all zeros -> all column will be zeros
        if first_row == 0:
            matrix[0] = [0]*m

                        

# class Solution:
#     def setZeroes(self, matrix: List[List[int]]) -> None:
#         """
#         Do not return anything, modify matrix in-place instead.
    
#         """
#         n, m = len(matrix), len(matrix[0])
#         row = set()
#         col = set()
#         for i in range(n):
#             for j in range(m):
#                 if matrix[i][j]==0:
#                     if i not in row and j not in col:
#                         col.add(j)
#                         row.add(i)
#                     elif i in row:
#                         col.add(j)
#                     elif j in col:
#                         row.add(i)
                        

#         for i in row:
#             matrix[i] = [0]*m
#         for j in col:
#             for i in range(n):
#                 matrix[i][j]=0
         

# O(m+n) extra space for counting each row and col + O(nm * 2) time for finding the answer: matrix + hash table
class Solution:    
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        cols = [0] *len(picture[0])
        rows = [0] * len(picture)
        res = 0
        for i in range(len(picture)):
            for j in range(len(picture[0])):
                if picture[i][j] == "B":
                    cols[j]+=1
                    rows[i]+=1
        for i in range(len(picture)):
            for j in range(len(picture[0])):
                if picture[i][j] == "B":
                    if cols[j]==1 and rows[i]==1:
                        res+=1
                    
        return res