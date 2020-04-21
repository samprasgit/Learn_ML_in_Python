def NumberOF1(n):
    count=0
    if n<0:
        n=n&0xffffffff

    while n:
        count+=1
        n=n&(n-1)
    return count
