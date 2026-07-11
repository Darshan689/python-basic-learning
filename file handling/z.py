# peffect binary tree:
#class for perfect bnt:
# class binary_tree_node:
#     def __init__(self,data):
#         self.data = data
#         self.left = None
#         self.right =None
# # object creation for perfect binary tree:

# btn1 = binary_tree_node(1)
# btn2 = binary_tree_node(2)
# btn3 =  binary_tree_node(3)
# btn4 =  binary_tree_node(4)
# btn5 =  binary_tree_node(5)
# btn6 =  binary_tree_node(6)
# btn7 =  binary_tree_node(7)

# #applyin operations:

# btn1.left = btn2
# btn1.right = btn3
# btn2.left = btn4
# btn2.right = btn5
# btn3.left = btn6
# btn3.rigth = btn7

class BINARY_TREE_NODE:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


# Create 3 nodes
BTN1 = BINARY_TREE_NODE(1)
BTN2 = BINARY_TREE_NODE(2)
BTN3 = BINARY_TREE_NODE(3)

# Connect nodes
BTN1.left = BTN2
BTN1.right = BTN3

# Print values
print("Root node:", BTN1.data)
print("Left child:", BTN1.left.data)
print("Right child:", BTN1.right.data)

# Check whether BTN1.left and BTN2 are the same object
print("Is BTN1.left connected to BTN2?", BTN1.left is BTN2)    




 