# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   690 员工的重要性.py
@Time    :   2021/05/01 23:06:29
@Desc    :   None
"""


class Employee:
    def __init__(self, id, importance, subordinates):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates


class Solution:

    def getImportance(self, employees, id):

        mp = {employees.id: employees for employee in employees}

        def dfs(id):
            employee = mp[id]
            total = employee.importance + \
                sum(dfs(subIdx) for subIdx in employee.subordinates)
            return total
        return dfs(id)


S = Solution()
employees, idx = Employee([1, 5, [2, 3]], [2, 3, []], [3, 3, []]), 1
print(S.getImportance(employees, id))
