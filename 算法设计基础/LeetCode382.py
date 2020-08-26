import random


class Solution:

    def __init__(self, head):
        self.head = head

    def getRandom(self):
        count = 0
        reverse = 0
        cur = self.head
        while cur:
            count += 1
            rand = random.randint(1, count)
            if rand == count:
                reverse = cur.val
            cur = cur.next
        return reverse
