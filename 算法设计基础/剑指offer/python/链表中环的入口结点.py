class Solution:

    def EntryNodeOfLoop(self, pHead):
        if pHead == None or pHead.next == None:
            return None
        p1 = pHead
        p2 = pHead
        while p1 and p2.next:
            p1 = p1.next
            p2 = p2.next
            if p1 == p2:
                p1 = pHead
                while p1 != p2:
                    p1 = p1.next
                    p2 = p2.next
                return p1
        return None
