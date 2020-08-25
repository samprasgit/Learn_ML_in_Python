<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [分治](#%E5%88%86%E6%B2%BB)
  - [引文](#%E5%BC%95%E6%96%87)
  - [主要思想](#%E4%B8%BB%E8%A6%81%E6%80%9D%E6%83%B3)
  - [分治算法的步骤](#%E5%88%86%E6%B2%BB%E7%AE%97%E6%B3%95%E7%9A%84%E6%AD%A5%E9%AA%A4)
  - [分治法适用的情况](#%E5%88%86%E6%B2%BB%E6%B3%95%E9%80%82%E7%94%A8%E7%9A%84%E6%83%85%E5%86%B5)
  - [伪代码](#%E4%BC%AA%E4%BB%A3%E7%A0%81)
  - [复杂度分析](#%E5%A4%8D%E6%9D%82%E5%BA%A6%E5%88%86%E6%9E%90)
  - [举个栗子](#%E4%B8%BE%E4%B8%AA%E6%A0%97%E5%AD%90)
    - [数组逆序对计算](#%E6%95%B0%E7%BB%84%E9%80%86%E5%BA%8F%E5%AF%B9%E8%AE%A1%E7%AE%97)
  - [算法应用](#%E7%AE%97%E6%B3%95%E5%BA%94%E7%94%A8)
    - [169. 多数元素](#169-%E5%A4%9A%E6%95%B0%E5%85%83%E7%B4%A0)
    - [4.两个排序数组的中位数](#4%E4%B8%A4%E4%B8%AA%E6%8E%92%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E4%B8%AD%E4%BD%8D%E6%95%B0)
    - [53. 最大子序和](#53-%E6%9C%80%E5%A4%A7%E5%AD%90%E5%BA%8F%E5%92%8C)
    - [50. Pow(x, n)](#50-powx-n)
    - [241. 为运算表达式设计优先级](#241-%E4%B8%BA%E8%BF%90%E7%AE%97%E8%A1%A8%E8%BE%BE%E5%BC%8F%E8%AE%BE%E8%AE%A1%E4%BC%98%E5%85%88%E7%BA%A7)
  - [参考资料](#%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

##  分治

### 引文

MapReduce（分治算法的应用） 是 Google 大数据处理的三驾马车之一，另外两个是 GFS 和 Bigtable。它在倒排索引、PageRank 计算、网页分析等搜索引擎相关的技术中都有大量的应用。

尽管开发一个 MapReduce 看起来很高深，感觉遥不可及。实际上，万变不离其宗，它的本质就是分治算法思想，分治算法。如何理解分治算法？为什么说 MapRedue 的本质就是分治算法呢？

### 主要思想

分治算法的主要思想是将原问题**递归地分成**若干个子问题，直到子问题**满足边界条件**，停止递归。将子问题逐个击破(一般是同种方法)，将已经解决的子问题合并，最后，算法会**层层合并**得到原问题的答案。

### 分治算法的步骤

* 分：**递归地**将问题**分解**为各个的子**问题**(性质相同的、相互独立的子问题)；
* 治：将这些规模更小的子问题**逐个击破**；
* 合：将已解决的子问题**逐层合并**，最终得出原问题的解；

![](https://img-blog.csdnimg.cn/20200408204450701.png)

### 分治法适用的情况

分治法所能解决的问题一般具有以下几个特征：

  1) 该问题的规模缩小到一定的程度就可以容易地解决

  2) 该问题可以分解为若干个规模较小的相同问题，即该问题具有最优子结构性质。

  3) 利用该问题分解出的子问题的解可以合并为该问题的解；

  4) 该问题所分解出的各个子问题是相互独立的，即子问题之间不包含公共的子子问题。

第一条特征是绝大多数问题都可以满足的，因为问题的计算复杂性一般是随着问题规模的增加而增加；

**第二条特征是应用分治法的前提**它也是大多数问题可以满足的，此特征反映了递归思想的应用；、

**第三条特征是关键，能否利用分治法完全取决于问题是否具有第三条特征**，如果**具备了第一条和第二条特征，而不具备第三条特征，则可以考虑用贪心法或动态规划法**。

**第四条特征涉及到分治法的效率**，如果各子问题是不独立的则分治法要做许多不必要的工作，重复地解公共的子问题，此时虽然可用分治法，但**一般用动态规划法较好**。



### 伪代码

```python
def divide_conquer(problem, paraml, param2,...):
    # 不断切分的终止条件
    if problem is None:
        print_result
        return
    # 准备数据
    data=prepare_data(problem)
    # 将大问题拆分为小问题
    subproblems=split_problem(problem, data)
    # 处理小问题，得到子结果
    subresult1=self.divide_conquer(subproblems[0],p1,..…)
    subresult2=self.divide_conquer(subproblems[1],p1,...)
    subresult3=self.divide_conquer(subproblems[2],p1,.…)
    # 对子结果进行合并 得到最终结果
    result=process_result(subresult1, subresult2, subresult3,...)
```

### 复杂度分析

- 时间复杂度

  $O(Nlog())$

### 举个栗子

#### 数组逆序对计算

​		通过应用举例分析理解分治算法的原理其实并不难，但是要想灵活应用并在编程中体现这种思想中却并不容易。所以，这里这里用分治算法应用在排序的时候的一个栗子，加深对分治算法的理解。

相关概念：

- **有序度**：表示一组数据的有序程度
- **逆序度**：表示一组数据的无序程度

一般通过**计算有序对或者逆序对的个数**，来表示数据的有序度或逆序度。

假设我们有 `n` 个数据，我们期望数据从小到大排列，那完全有序的数据的有序度就是 $n(n-1)/2$，逆序度等于 0；相反，倒序排列的数据的有序度就是 0，逆序度是 $n(n-1)/2$。

**Q：如何编程求出一组数据的有序对个数或者逆序对个数呢？**

因为有序对个数和逆序对个数的求解方式是类似的，所以这里可以只思考逆序对（常接触的）个数的求解方法。

- 方法1
  - 拿数组里的每个数字跟它后面的数字比较，看有几个比它小的。
  
  - 把比它小的数字个数记作 `k`，通过这样的方式，把每个数字都考察一遍之后，然后对每个数字对应的 `k` 值求和
  
  - 最后得到的总和就是逆序对个数。
  
  - 这样操作的时间复杂度是$O(n^2)$（需要两层循环过滤）。那有没有更加高效的处理方法呢？这里尝试套用分治的思想来求数组 A 的逆序对个数。
  
    ```python
    def inversePairs(nums):
        n = len(nums)
        if n < 2:
            return 0
        ans = 0
        for i in range(n):
            for j in range(i):
                if nums[j] > nums[i]:
                    ans += 1
    
        return ans
    ```
  
- 方法2
  - 首先将数组分成前后两半 A1 和 A2，分别计算 A1 和 A2 的逆序对个数 K1 和 K2

  - 然后再计算 A1 与 A2 之间的逆序对个数 K3。那数组 A 的逆序对个数就等于 K1+K2+K3。

  - 注意使用分治算法其中一个要求是，**子问题合并的代价不能太大**，否则就起不了降低时间复杂度的效果了。:star:

  - **如何快速计算出两个子问题 A1 与 A2 之间的逆序对个数呢？这里就要借助归并排序算法了。（这里先回顾一下归并排序思想）**如何借助归并排序算法来解决呢？归并排序中有一个非常关键的操作，就是将两个有序的小数组，合并成一个有序的数组。实际上，在这个合并的过程中，可以计算这两个小数组的逆序对个数了。每次合并操作，我们都计算逆序对个数，把这些计算出来的逆序对个数求和，就是这个数组的逆序对个数了。

  - 计算横跨两个区间的逆序对

    ![image.png](https://pic.leetcode-cn.com/0adb9d76f0f2a8efccaa1c3d340003e91e2a9eb9dc490280460acae0c8850a24-image.png)

    ![image.png](https://pic.leetcode-cn.com/a13af31b7f9e12f6d8588d95dd71c94aa0117bc8c819899e7806a5695e237f78-image.png)

    即在 `j` 指向的元素赋值回去的时候，给计数器加上 `mid - i + 1`

    - 在第 2 个子区间元素归并回去的时候，计算逆序对的个数，即后有序数组中元素出列的时候，计算逆序个数

      ```python
      from typing import List
      
      
      class Solution2:
      	'''
      	在第 2 个子区间元素归并回去的时候，计算逆序对的个数
      
      	'''
      
          def reversePairs(self,nums):
              n = len(nums)
              if n < 2:
                  return 0
              # 构造辅助数组
              tmp = [0 for _ in range(n)]
              return self.reversePairs(nums, 0, n - 1, tmp)
      
          def count_reverse_pairs(self, numsleft, right, tmp):
              # 统计数组nums区间[left,right]的逆序对
              if left == right:
                  return 0
              mid = (left + right) // 2
              left_pairs = self.count_reverse_pairs(nums, left, mid, tmp)
              right_pairs = self.count_reverse_pairs(nums, mid + 1, right, tmp)
              reverse_pairs = left_pairs + right_pairs
              if nums[mid] <= nims[right]:
                  return reverse_pairs
      
              reverse_cross_pairs = self.merge_and_count(nums, left, mid, right, tmp)
      
              return reverse_pairs + reverse_cross_pairs
      
          def merge_and_count(self, nums, left, mid, right, tmp):
              """
              [left, mid] 有序，[mid + 1, right] 有序
      
              前：[2, 3, 5, 8]，后：[4, 6, 7, 12]
              只在后面数组元素出列的时候，数一数前面这个数组还剩下多少个数字，
              由于"前"数组和"后"数组都有序，
              此时"前"数组剩下的元素个数 mid - i + 1 就是与"后"数组元素出列的这个元素构成的逆序对个数
      
              """
      
              for i in range(left, right + 1):
                  tmp[i] = nums[i]
      
              i = left
              j = mid + 1
              res = 0
              for k in range(left, right + 1):
                  if i > mid:
                      nums[k] = tmp[j]
                      j += 1
                  elif j > right:
                      nums[k] = tmp[i]
                      i += 1
                  elif tmp[i] <= tmp[j]:
                      # 此时前数组元素出列
                      nums[k] = tmp[i]
                      i += 1
                  else:
                      # 此时后数组元素出列 同喜逆序对数
                      nums[k] = tmp[j]
                      j += 1
                      res += (mid - i + 1)
              return res
        
      ```

      

    - 在第 1 个子区间元素归并回去的时候，计算逆序对的个数

      ```python
      class Solution3:
      	"""
      	在第 2 个子区间元素归并回去的时候，计算逆序对的个数
      	
      	"""  
      	def reversePairs(self, nums):
      		n=len(nums)
      		if n<2:
      			return 0  
      		# 构造辅助数组 
      		tmp=[0 for _ in range(n)]  
      		return self.reverse_pairs(nums,0,n-1,tmp)   
      	def reverse_pairs(self, nums,left,right,tmp):
      		if left==right:
      			return 0   
      		mid=(left+right)//2
      		left_paris=self.reverse_pairs(nums,left,mid,tmp)
      		right_paris=self.reverse_pairs(nums,mid+1,right,tmp)
      
      		reverse_pairs=left_paris+right_paris
      		if nums[mid]<=nums[mid+1]:
      			return reverse_pairs
      		reverse_cross_pairs=self.merge_and_count(nums,left,mid,right,tmp)
      		return reverse_pairs+reverse_cross_pairs
      
      	def merge_and_count(self,nums,left,mid,right,tmp):
      
      		for i in range(left, right + 1):
      			tmp[i]=nums[i]
      
      		i=left
      		j=mid+1 
      
      		res=0 
      		for k in range(left,right + 1):
      			if i>mid:
      				nums[k] = tmp[j]
      				j+=1 
      			elif j>right:
      				nums[k] = tmp[i]
      				i+=1 
      			elif tmp[i] <= tmp[j]:
      				nums[k] = tmp[j]
      				i+=1    
      				res+=(j-mid-1)
      
      			else: 
      				assert temp[i] > temp[j]  
      				nums[k] =tmp[j]
      				j+=1   
      
      		return res
      ```

      

  - 复杂度分析

    时间复杂度：$O(Nlog(N))$

    空间复杂度：$O(N)$

### 算法应用

#### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

* 题目描述

  给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 [n/2] 的元素。

  你可以假设数组是非空的，并且给定的数组总是存在众数。

  示例 1:

  ```python
  输入: [3,2,3]
  输出: 3
  ```

  示例 2:

  ```python
  输入: [2,2,1,1,1,2,2]
  输出: 2
  ```

* 解题思路

  * 确定切分的终止条件
  
    直到所有的子问题都是长度为 1 的数组，停止切分。
  
  * 准备数据，将大问题切分为小问题
  
    递归地将原数组二分为左区间与右区间，直到最终的数组只剩下一个元素，将其返回
  
  * 处理子问题得到子结果，并合并
  
    - 长度为 1 的子数组中唯一的数显然是众数，直接返回即可。
  
    - 如果它们的众数相同，那么显然这一段区间的众数是它们相同的值。
  
    - 如果他们的众数不同，比较两个众数在整个区间内出现的次数来决定该区间的众数

* 代码

  ```python
  class Solution(object):
      def majorityElement2(self, nums):
          """
          :type nums: List[int]
          :rtype: int
          """
          # 【不断切分的终止条件】
          if not nums:
              return None
          if len(nums) == 1:
              return nums[0]
          # 【准备数据，并将大问题拆分为小问题】
          left = self.majorityElement(nums[:len(nums)//2])
          right = self.majorityElement(nums[len(nums)//2:])
          # 【处理子问题，得到子结果】
          # 【对子结果进行合并 得到最终结果】
          if left == right:
              return left
          if nums.count(left) > nums.count(right):
              return left
          else:
              return right    
  ```
  

#### [4.两个排序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

- 题目描述

  两个排序的数组A和B分别含有m和n个数，找到两个排序数组的中位数

  示例

  ```
  给出数组A = [1,2,3,4,5,6] B = [2,3,4,5]，中位数3.5
  给出数组A = [1,2,3] B = [4,5]，中位数 3
  ```

- 解题思路

  median =（max（left_part）+ min（right_part））/ 2

  

  

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

* 题目描述

  给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

  示例:

  ```
  输入: [-2,1,-3,4,-1,2,1,-5,4],
  输出: 6
  解释: 连续子数组 [4,-1,2,1] 的和最大为6。
  ```

* 解题思路

  * 确定切分的终止条件

    直到所有的子问题都是长度为 1 的数组，停止切分。

  * 准备数据，将大问题切分为小问题

    递归地将原数组二分为左区间与右区间，直到最终的数组只剩下一个元素，将其返回

  * 处理子问题得到子结果，并合并

    - 将数组切分为左右区间
      - 对与左区间：从右到左计算左边的最大子序和
      - 对与右区间：从左到右计算右边的最大子序和

    - 由于左右区间计算累加和的方向不一致，因此，左右区间直接合并相加之后就是整个区间的和
    - 最终返回左区间的元素、右区间的元素、以及整个区间(相对子问题)和的最大值

* 代码

  ```python
  class Solution(object):
      def maxSubArray(self, nums):
          """
          :type nums: List[int]
          :rtype: int
          """
          # 【确定不断切分的终止条件】
          n = len(nums)
          if n == 1:
              return nums[0]
  
          # 【准备数据，并将大问题拆分为小的问题】
          left = self.maxSubArray(nums[:len(nums)//2])
          right = self.maxSubArray(nums[len(nums)//2:])
  
          # 【处理小问题，得到子结果】
          #　从右到左计算左边的最大子序和
          max_l = nums[len(nums)//2 -1] # max_l为该数组的最右边的元素
          tmp = 0 # tmp用来记录连续子数组的和
          
          for i in range( len(nums)//2-1 , -1 , -1 ):# 从右到左遍历数组的元素
              tmp += nums[i]
              max_l = max(tmp ,max_l)
              
          # 从左到右计算右边的最大子序和
          max_r = nums[len(nums)//2]
          tmp = 0
          for i in range(len(nums)//2,len(nums)):
              tmp += nums[i]
              max_r = max(tmp,max_r)
              
          # 【对子结果进行合并 得到最终结果】
          # 返回三个中的最大值
          return max(left,right,max_l+ max_r)
  ```

  

#### [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

* 题目描述

  实现 `pow(x, n) `，即计算 `x` 的 `n` 次幂函数。

  示例 1:

  ```
  输入: 2.00000, 10
  输出: 1024.00000
  ```

  示例 2:

  ```
  输入: 2.10000, 3
  输出: 9.26100
  ```

  示例 3:

  ```
  输入: 2.00000, -2
  输出: 0.25000
  解释: 2-2 = 1/22 = 1/4 = 0.25
  ```

  说明:

  `-100.0 < x < 100.0`
  `n `是 32 位有符号整数，其数值范围是$[−2^{31}, 2^{31} − 1] $。

* 解题思路

  * 确定切分的终止条件

    对`n`不断除以2，并更新`n`，直到为0，终止切分

  * 准备数据，将大问题切分为小问题

    对`n`不断除以2，更新

  * 处理子问题得到子结果，并合并

    * `x`与自身相乘更新`x`
    * 如果`n%2 ==1`
      - 将`p`乘以`x`之后赋值给`p`(初始值为1)，返回`p`
  * 最终返回`p`

* 代码

  ```python
  class Solution(object):
      def myPow(self, x, n):
          """
          :type x: float
          :type n: int
          :rtype: float
          """
          # 处理n为负的情况
          if n < 0 :
              x = 1/x
              n = -n
          # 【确定不断切分的终止条件】
          if n == 0 :
              return 1
  
          # 【准备数据，并将大问题拆分为小的问题】
          if n%2 ==1:
            # 【处理小问题，得到子结果】
            p = x * self.myPow(x,n-1)# 【对子结果进行合并 得到最终结果】
            return p
          return self.myPow(x*x,n/2) 
  ```



将k个排序好的链表合并成新的有序链表

#### [241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)

### 参考资料

[五大常用算法之一：分治算法](https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741370.html)

[你不知道的 Python 分治算法](https://gitchat.csdn.net/activity/5d39b22ba2a28a54fc0090f7?utm_source=so#1?utm_source=so&utm_source=so)

[分治算法案例— 两个排序数组的中位数](https://blog.csdn.net/chenvast/article/details/78949992)



