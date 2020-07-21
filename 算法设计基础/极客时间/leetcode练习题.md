数组、链表

### 2.[删除数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array)

给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

```
给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

你不需要考虑数组中超出新长度后面的元素。
```

- 双指针法

  ```
  数组完成排序后，我们可以放置两个指针 i 和 j，其中 i 是慢指针，而 j 是快指针。只要 nums[i] = nums[j]，我们就增加 j 以跳过重复项。
  
  当我们遇到 nums[j]！=nums[i]时，跳过重复项的运行已经结束，因此我们必须把它（nums[j]）的值复制到 nums[i + 1]。然后递增 i，接着我们将再次重复相同的过程，直到 j 到达数组的末尾为止
  
  ```

  ![1.png](https://pic.leetcode-cn.com/0039d16b169059e8e7f998c618b6c2b269c2d95b02f43415350bde1f661e503a-1.png)

  ```python
  def removeDuplicates(nums):
  	'''
  	
  	双指针法  
  	时间复杂度：O(N)
  	空间复杂度:  O（1）
  	
  	Arguments:
  		nums {[type]} -- [description]
  	
  	Returns:
  		number -- [description]
  	'''
      if len(nums) < 1:
          return 0
      i = 0
      for j in range(len(nums)):
          if (nums[j] != nums[i]):
              i += 1
              nums[i] = nums[j]
  
      return i + 1
  ```

  - 优化

    ![2.png](https://pic.leetcode-cn.com/06e80bea0bfa0dadc6891407a237fef245f950cab74d050027ac3beecb65d778-2.png)

    此时数组中没有重复元素，按照上面的方法，每次比较时 nums[p]都不等于 nums[q]，因此就会将 `q` 指向的元素原地复制一遍，这个操作其实是不必要的。

    因此我们可以添加一个小判断，当 `q - p > 1` 时，才进行复制。

    

