
# pyhton内置函数int() bin


def addBinary(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]


a, b = "100", "100"
print(addBinary(a, b))
