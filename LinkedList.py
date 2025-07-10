

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
      newNode = Node(data)
      if self.head is None:
        self.head = newNode
        return
      theNode = self.head  
      while theNode.next is not None:
        theNode = theNode.next
      theNode.next = newNode

    def print_list(self):
      cur_node = self.head
      while cur_node:
          print(cur_node.data)
          cur_node = cur_node.next

# Example Usage:
llist = LinkedList()
llist.append("John Doe")
llist.append("Jane Smith")
llist.print_list()
