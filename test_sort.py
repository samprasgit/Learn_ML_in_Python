def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))

    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0))

    return result


def mergeSort(arry):
    n = len(arry)
    if n < 2:
        return arry
    mid = n // 2
    left, right = arry[0: mid], arry[mid:]

    return merge(mergeSort(left), mergeSort(right))

nums = [2, 3, 5, 7, 1, 4, 6, 8]
print("排序之前：", nums)
print("排序之后：", mergeSort(nums))
