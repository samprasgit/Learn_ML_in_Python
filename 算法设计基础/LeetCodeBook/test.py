nums = [2, 7, 11, 15]


def permutations(arr, position, end):

    if position == end:
        print(arr)

    else:
        for index in range(position, end):

            arr[index], arr[position] = arr[position], arr[index]
            permutations(arr, position + 1, end)
            arr[index], arr[position] = arr[position], arr[index]


print(permutations(nums, 0, 4))
