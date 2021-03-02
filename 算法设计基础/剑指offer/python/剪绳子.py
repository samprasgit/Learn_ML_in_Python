class Solution:
    def cutttingRope(self, n):
        from math import pow
        if n <= 3: return n - 1
        a, b = n // 3, n % 3
        if b == 0: return int(pow(3, a))
        if b == 1: return int(pow(3, a - 1) * 4)
        return int(pow(3, a) * 2)


class Solution2:
    def cuttingRope(self, n):
        """
        大数越界
        """
        if n <= 3:
            return n - 1
        a, b, p, x, rem = n // 3 - 1, n % 3, 1000000007, 3, 1
        while a > 0:
            if a % 2: rem = (rem * x) % p
            x = x ** 2 % p
            a //= 2

        if b == 0: return (rem * 3) % p
        if b == 1: return (rem * 4) % p
        return (rem * 6) % p
