<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [题目](#%E9%A2%98%E7%9B%AE)
- [解法一：排序算法](#%E8%A7%A3%E6%B3%95%E4%B8%80%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95)
- [解法二：最小堆](#%E8%A7%A3%E6%B3%95%E4%BA%8C%E6%9C%80%E5%B0%8F%E5%A0%86)
- [解法三：桶排序](#%E8%A7%A3%E6%B3%95%E4%B8%89%E6%A1%B6%E6%8E%92%E5%BA%8F)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

### 题目

给定一个非空的整数数组，返回其中出现频率前 **k** 高的元素。

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 1:**

```
输入: nums = [1], k = 1
输出: [1]
```

> 时间复杂度必须优于$O(nlog(n))$

### 解法一：排序算法

- #### 思路

- #### python实现

- #### 复杂度分析

### 解法二：最小堆

- #### 思路

  > 最终需要返回前 k 个频率最大的元素，可以想到借助堆这种数据结构。通过维护一个元素数目为 k的最小堆，每次都将新的元素与堆顶端的元素（堆中频率最小的元素）进行比较，如果新的元素的频率比堆顶端的元素大，则弹出堆顶端的元素，将新的元素添加进堆中。最终，堆中的 k 个元素即为前 k 个高频元素

- #### python实现

  ```python
  class Solution:
      def topKFrequent(self, nums, k):
          """
          :type nums: List[int]
          :type k: int
          :rtype: List[int]
          """
          # 统计元素的频率
          freq_dict = dict()
          for num in nums:
              freq_dict[num] = freq_dict.get(num, 0) + 1
              
          # 维护一个大小为k的最小堆，使得堆中的元素即为前k个高频元素
          pq = list()
          for key, value in freq_dict.items():
              if len(pq) < k:
                  heapq.heappush(pq, (value, key))
              elif value > pq[0][0]:
                  heapq.heapreplace(pq, (value, key))
                  
          # 取出堆中的元素
          ret = list()
          while pq:
              ret.append(heapq.heappop(pq)[1])
          return ret
  ```

  

- #### 复杂度分析

### 解法三：桶排序

- #### 思路

  > 为了进一步优化时间复杂度，可以采用桶排序（bucket sort），即用空间复杂度换取时间复杂度

- #### python实现

  ```python
  class Solution:
      def topKFrequent(self, nums, k):
          """
          :type nums: List[int]
          :type k: int
          :rtype: List[int]
          """
          # 统计元素的频率
          freq_dict = dict()
          for num in nums:
              freq_dict[num] = freq_dict.get(num, 0) + 1
  
          # 桶排序
          bucket = [[] for _ in range(len(nums) + 1)]
          for key, value in freq_dict.items():
              bucket[value].append(key)
  
          # 逆序取出前k个元素
          ret = list()
          for i in range(len(nums), -1, -1):
              if bucket[i]:
                  ret.extend(bucket[i])
              if len(ret) >= k:
                  break
          return ret[:k]
  ```

  

- #### 复杂度分析

  时间复杂度:$O(n)$

  空间复杂度:$O(n)$

