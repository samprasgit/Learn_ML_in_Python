N = input()


def clib(N):
    if N <= 3:
        return N
    else:
        for _ in range(4, N + 1):
            return clib(N - 1) + clib(N - 2)


def stairs(N):
    if N <= 3:
        return N
    else:
        return clib(N) - 2**(N - 4)

print(stairs(N))
