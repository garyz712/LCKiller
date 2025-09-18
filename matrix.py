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
         