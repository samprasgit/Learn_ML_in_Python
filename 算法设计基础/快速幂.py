def power(base, exponent):
    res = 1
    while exponent:
        if exponent & 1:
            res *= base
        base *= base
        exponent = exponent >> 1
    return res
