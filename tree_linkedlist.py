#✅ In-order (Left → Root → Right)
def inorder(node):
    if not node:
        return
    inorder(node.left)
    print(node.val)
    inorder(node.right)

#✅ Pre-order (Root → Left → Right)
def preorder(node):
    if not node:
        return
    print(node.val)
    preorder(node.left)
    preorder(node.right)

#✅ Post-order (Left → Right → Root)
def postorder(node):
    if not node:
        return
    postorder(node.left)
    postorder(node.right)
    print(node.val)


  #DFS with stack + In/Pre/Post order traversal without recursion
from typing import Optional, Literal

class BSTIterator:
    def __init__(self, root: Optional['TreeNode'], mode: Literal["in", "pre", "post"] = "in"):
        """
        mode = "in"   -> inorder   (left, root, right)
        mode = "pre"  -> preorder  (root, left, right)
        mode = "post" -> postorder (left, right, root)
        """
        self.mode = mode
        self.stack = []
        
        if mode == "in":
            # push all the way left
            while root:
                self.stack.append(root)
                root = root.left

        elif mode == "pre":
            if root:
                self.stack.append(root)

        elif mode == "post":
            if root:
                self.stack.append((root, False))  # (node, visited)

    def next(self) -> int:
        if self.mode == "in":
            node = self.stack.pop()
            curr = node.right
            while curr:
                self.stack.append(curr)
                curr = curr.left
            return node.val

        elif self.mode == "pre":
            node = self.stack.pop()
            if node.right:
                self.stack.append(node.right)
            if node.left:
                self.stack.append(node.left)
            return node.val

        elif self.mode == "post":
            while self.stack:
                node, visited = self.stack.pop()
                if visited:
                    return node.val
                # push node back as visited, then its children
                self.stack.append((node, True))
                if node.right:
                    self.stack.append((node.right, False))
                if node.left:
                    self.stack.append((node.left, False))
            raise StopIteration("No more elements")

    def hasNext(self) -> bool:
        return bool(self.stack)


# In order
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        self.prev = None          # previous value visited in-order
        self.ans  = float("inf")  # current best gap
        
        def dfs(node):
            if not node:                              # base case
                return
            dfs(node.left)                            # left
            if self.prev is not None:                 # visit
                self.ans = min(self.ans, node.val - self.prev)
            self.prev = node.val
            dfs(node.right)                           # right
        
        dfs(root)
        return self.ans

# recursively travel two trees at the same time
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def isMirror(t1, t2):
            if not t1 and not t2:
                return True
            if not t1 or not t2:
                return False
            return (t1.val == t2.val and
                    isMirror(t1.left, t2.right) and
                    isMirror(t1.right, t2.left))
        
        return isMirror(root.left, root.right)
        
# BFS for binary tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if not root:
            return []
        
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level_sum = 0

            for i in range(level_size): #process one layer at a time because one node only can have two childs
                node = queue.popleft()
                level_sum += node.val

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level_sum / level_size)

        return result

        

# DFS + Dynamic programming: At each node, I can "choose" whether to include left, right, both, or none, but I must return only one path upward — similar to decision making in DP.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')
        
        def dfs(node):
            if not node:
                return 0
            
            # Only keep positive gains
            left_gain = max(dfs(node.left), 0)
            right_gain = max(dfs(node.right), 0)
            
            # Path with highest sum through this node
            current_max_path = node.val + left_gain + right_gain
            
            # Update global max with node plitting
            self.max_sum = max(self.max_sum, current_max_path)
            
            # Return max gain if continuing upward without splitting
            return node.val + max(left_gain, right_gain)
        
        dfs(root)
        return self.max_sum



#Linked list + Hash map/in place modification
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList1(self, head: 'Optional[Node]') -> 'Optional[Node]':
        old_to_new = {}

        # First pass: copy all nodes
        curr = head
        while curr:
            old_to_new[curr] = Node(curr.val)
            curr = curr.next

        # Second pass: assign next and random
        curr = head
        while curr:
            old_to_new[curr].next = old_to_new.get(curr.next)
            old_to_new[curr].random = old_to_new.get(curr.random)
            curr = curr.next

        return old_to_new.get(head)
    
    #O(1) space
    # Step 1: Clone each node and insert it right after the original
    # Original:
    # A → B → C
    # After step 1:
    # A → A' → B → B' → C → C'

    # Step 2: Set the random pointers for the cloned nodes
    # If original.random = X, then original.next.random = original.random.next

    # Step 3: Separate the original and cloned lists

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None

        # Step 1: Insert cloned nodes next to originals
        curr = head
        while curr:
            copy = Node(curr.val, curr.next)
            curr.next = copy
            curr = copy.next

        # Step 2: Assign random pointers to copied nodes
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next

        # Step 3: Separate the lists
        curr = head
        copy_head = head.next
        while curr:
            copy = curr.next
            curr.next = copy.next
            if copy.next:
                copy.next = copy.next.next
            curr = curr.next

        return copy_head
                


# reversing linked list
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # Step 0: Edge case
        if not head or left == right:
            return head
        
        # Step 1: Dummy node to handle head reversal: if left=0 or return head
        dummy = ListNode(0)
        dummy.next = head
        
        # Step 2: Move `leftPrev` to the node before the "left" node
        leftPrev = dummy
        for _ in range(left - 1):
            leftPrev = leftPrev.next
        curr = leftPrev.next         # This will be the tail of the reversed sublist
        
        # Step 3: Reverse the sublist [left, right] using 3 pointers      
        prev = None # set the first back connection to None because it is not visited yet
        for _ in range(right - left + 1):
            next_temp = curr.next # save the curr.next first for later curr update
            curr.next = prev # reconnect curr's next to the one before it
            prev = curr # update back connection node to be curr
            curr = next_temp # udpate curr as saved
        
        # Step 4: Reconnect the reversed sublist back to original list
        leftPrev.next.next = curr # Connect tail of reversed part to node after right through the old leftPrev
        leftPrev.next = prev  # Update the leftPrev to connect to the new head of reversed part (prev)
        
        return dummy.next # will be the original head
        

    # Head insertion solution: cleaner and smarter
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        # Step 1: Move prev to the node just before the reversal
        for _ in range(left - 1):
            prev = prev.next
        curr = prev.next #curr and prev are never changed but curr will gradually move to the tail

        # Step 2: Reverse the sublist between left and right       
        for _ in range(right - left):
            temp = curr.next # save the curr.next first for later curr update as before
            curr.next = temp.next # connect curr to the current tail's next
            temp.next = prev.next # connect the new head to the current head (prev.next)
            prev.next = temp #update the current head but not the one before it (prev)

        return dummy.next



# Building Binary Tree from Inorder and Postorder
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder: #if nothing is on the left or right of the previous root, return None
            return None
        root = TreeNode(postorder.pop()) #last value in postorder must be root

        idx = inorder.index(root.val) #find all values on the left/right of the root

        root.right = self.buildTree(inorder[idx+1:], postorder) #must build rhs first due to poping from postorder

        root.left = self.buildTree(inorder[:idx], postorder) #build left after popping all the right recursively

        return root


# need a node before the group and anther node after the group; also how to update them
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        curr= head
        length=0
        while curr:
            curr = curr.next
            length+=1

        iterations = length//k

        dummy = ListNode(0, head)
        curr = head
        nextGroup = head #nextgroup to connect with the first node
        prevGroup = dummy #prevgroup to connect with the last node

        for i in range(iterations):
            for j in range(k):
                nextGroup = nextGroup.next #set nextgroup to the right position

            #reversing the current group
            nextNode = nextGroup #nextnode to connect is nextgroup
            for j in range(k):
                if j==0:
                    first = curr #the first node in this group will become the prevGroup for the next group, so set it up

                oldnext = curr.next #save the current next, will be the next curr
                curr.next = nextNode #connect current next to the right nextNode
                nextNode = curr #curr will be the next nextNode                
                curr = oldnext #update curr

            prevGroup.next = nextNode #connect the previous group with the last node
            prevGroup = first#update the prevgroup with the first node

        return dummy.next


# Two additional dummy Linked List nodes
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        left, right = ListNode(), ListNode()
        leftP, rightP = left, right

        while head:
            if head.val<x:
                leftP.next = head
                leftP = leftP.next
            else:
                rightP.next= head
                rightP = rightP.next

            head= head.next

        leftP.next = right.next #connect left and right
        if rightP.next: #if the last right element is pointing to left, reset it
            rightP.next = None

        return left.next


# Binary tree + linked list + dfs with return!
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # since you need to connect the tail of the flatten left to the head to flatten right, you need to return the right Tail of the left tree
        def dfs(node):
            if node == None:
                return None
            leftT = dfs(node.left) #right tail of the left tree
            rightT = dfs(node.right) #right tail of the right tree

            if node.left:
                leftT.right = node.right
                node.right = node.left
                node.left = None

            return rightT or leftT or node

        dfs(root)


# selective return a certain node and propagate in DFS

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == None:
            return None
        
        if root.val in [p.val, q.val]:
            return root #if root is a target, there is no need to travel downward
            
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right: #if q and p are in both left and right, root must be the LCA
            return root
        else:#if q and p are in either left and right, either left or right, whichever is first reached will be returned
            return left or right #return whichever is available or None


# merge sort + linked list + find middle in linked list
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next # is also a pointer to the next object node

class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Base case
        if not head or not head.next:
            return head

        #standard way to find the mid node pointer
        mid, end = head, head.next#if there are only two nodes, this is the answer

        while end != None and end.next != None: #while end is not the last node (even) nor the final empty node (odd), keep moving the pointers 
            mid = mid.next #move mid pointer one step forward
            end = end.next.next #move end pointer two step forward, although end may not have .next.next
        #at the end, mid will be the middle node in odd length or the left node in the even length

        right = mid.next 
        mid.next = None
        left = head

        leftsorted = self.sortList(left)
        rightsorted = self.sortList(right)

        return self.merge(leftsorted, rightsorted)
        
    def merge(self, l1, l2):
        n = ListNode() #n is a pointer to this object class
        start = n #start point to the same Node object (starting node)

        while l1 and l2:
            if l1.val < l2.val:
                n.next = l1 # (1) n point to the object l1 points to
                l1 = l1.next #l1 now points to the next object
            else:
                n.next = l2
                l2 = l2.next
            n = n.next # set n to be the old l1/l2 input node, its next pointer will be modified later in (1)

        n.next = l1 or l2 #either l1 or l2 will be non empty after while loop
        return start.next #ignore the starting empty node

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object): # use two point to find if there is a cycle in linked list
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True


# Three pointers for linkedlist reversing
class Solution:
    def reverseListPrt(self, head: Optional[ListNode]) -> Optional[ListNode]:

        curr = head
        prev = None
        while curr:
            prt = curr.next
            curr.next = prev
            prev=curr
            curr = prt
        return prev
    # use recursion to reverse the linked list, return the new head!
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None:
            return head
        
        newHead = self.reverseList(head.next)
        head.next.next = head # connect the tail of the reversed sublist to the current head
        head.next = None #pointing the new tail to None instead of the list
        return newHead


# DFS + binary tree
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = 0 #store the final diameter answer
        def dfs(r): # return the height of the current root
            nonlocal res
            if r == None:
                return -1
            left = dfs(r.left) #get the height of the left node
            right = dfs(r.right)
        
            res = max(left + right + 2, res)

            return max(left, right) +1 #return its own height
        dfs(root)
        return res


# Reverse Linkedlist + Slow fast pointers
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return True

        # Find the end of first half and reverse second half.
        second_half_start = self.second_half_start(head)
        second_half_start = self.reverse_list(second_half_start)

        # Check whether or not there's a palindrome.
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        return result

    def second_half_start(self, head: ListNode) -> ListNode:
        slow, fast = head, head

        while fast and fast.next: # the slow pointer must be at the start of the second half: fast can be None
        # while fast.next and fast.next.next: # this is for getting the slow pointer at the end of the first half: fast cannot be None
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverse_list(self, head: ListNode) -> ListNode:
        previous = None
        current = head
        while current:
            temp = current.next
            current.next = previous
            previous = current
            current = temp
        return previous



# 3 pointers for cycle

# slow: x1+x2
# fast: x1+x2+x3+x2
# 2*slow =fast ->  x1=x3
# use pos pointer to meet slow pointer
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast, pos = head, head, head
        start = False
        while slow != fast or start == False:
            start = True
            if fast == None or fast.next == None:
                return None
            slow = slow.next
            fast = fast.next.next
            
        while pos != slow:
            slow = slow.next
            pos = pos.next

        return pos


# Not a DP problem, just updating the ans during DFS traversal

class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.ans =0
        def findlongest(root, currentDir, path):
            if root:
                self.ans = max(path, self.ans)

                if currentDir == "left":
                    findlongest(root.right, "right", path+1)
                    findlongest(root.left, "left", 1)
                else:
                    findlongest(root.left, "left", path+1)
                    findlongest(root.right, "right", 1)


        findlongest(root, "DoesNotMatter", 0)

        return self.ans