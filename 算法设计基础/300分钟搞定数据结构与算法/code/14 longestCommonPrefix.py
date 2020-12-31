# -*- coding: utf-8 -*-


def longestCommonPrefix(strs):

    if len(strs) == 0:
        return ""
    result = []
    index = 0
    minLen = 999
    for item in strs:
        if len(item) < minLen:
            minLen = len(item)
    while index < minLen:
        for item in strs:
            if strs[0][index] != item[index]:
                return "".join(result)

        result.append(strs[0][index])
        index += 1
    return "".join(result)


print(longestCommonPrefix(["flower", "flight", "fly"]))
