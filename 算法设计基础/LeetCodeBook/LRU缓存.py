# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   LRU缓存.py
@Time    :   2021/04/29 13:59:22
@Desc    :   None
"""
# LRU缓存
# 哈希表+双向链表


class DLinkNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.cache = dict()
        # 使用伪头部和伪尾部节点
        self.head = DLinkNode()
        self.tail = DLinkNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key):
        if key not in self.cache:
            return -1
        # 如果key存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key, value):
        if key not in self.cache:
            # key不存在，创建一个新的节点
            node = DLinkNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -= 1

        else:
            # key存在 先定位 在修改value 并移到头部
            node = self.cache[key]
            node.value = value
            self.removeToHead(node)

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, head):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self, node):
        node = self.tail.prev
        self.removeNode(node)
        return ndoe
