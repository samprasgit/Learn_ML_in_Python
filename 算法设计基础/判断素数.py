def prime_numbers(n):
    if n == 1:
        return False
    for i in range(2, n // 2 + 1):
        if n % i == 0:
            return False
        else:
            return True


num = []
for i in range(1000):
    if prime_numbers(i):
        num.append(i)

print(num)
