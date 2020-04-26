# !/usr/bin/env python
# -*- coding: utf-8 -*-


def findKthLargest0(nums, k):
	"""普通的排序方法"""
	nums.sort()
	return nums[-k]


def findKthLargest1(nums, k):
	"""堆排序"""


class maxheap:

	def __init__(self):
		self._data = []
		self._count = 0

	def size(self):
		return self._count

	def add(self, x):
		"""
        往最大堆中添加元素：
        1. 首先把新添加的元素当做最后一个叶子节点添加到堆的最后面
        2. 判断当前节点和其父节点的大小：若当前节点大于父节点，那么交换两个节点的位置；然后继续往上比较，直到当前节点小于其父节点（即在_shiftUp函数中实现的逻辑）
        :param x: 需要加入到最大堆中的元素
        :return: 返回调整后的最大堆

        """

        self._data.append(x)
        self._shift_up(self._count)
        self._count += 1

    def pop(self):
    	
    	"""
        从最大堆中弹出根节点，该元素肯定是这个堆中最大的元素；调整堆的结构使得新的堆仍是一个最大堆：
        1. 首先弹出根节点（也就是索引为0的元素），然后把最后一个叶子节点放到根节点位置
        2. 从根节点开始，比较其与两个子节点的大小：当当前节点小于其子节点时，则将当前节点与较大的一个子节点交换位置，然后继续往下比较，直到当前节点是叶子节点或者当前节点大于子节点
        :return: 返回调整后的最大堆
        """

        ret=self._data[0]
        self._count-=1
        self._data[0]=self._data[-1]
        self._shift_down(0)
        return ret

    def _shift_up(self,index):
    	parent=(index-1) >>1 
    	













