class Solution:

    def getKthGromEnd(self, had, k):
        fast = slow = head
        if herad == None or k == 0:
            return None
        while k > 0 and fast != None:
            fast = fast.next
            k -= 1

        while fast != None:
            fast = fast.next
            slow = slow.next
        return slow
