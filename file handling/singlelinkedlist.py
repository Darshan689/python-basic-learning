class Node:
    def __init__(self,data):
        self.data = data 
        self.next = None
class SLL:
    def __init__(self):
        self.head = None
x = SLL()
n1 = Node(10)
x.head = n1

n2 = Node(20)
n1.head = n2

n3 = Node(30)
n2.head = n3

print(x)
