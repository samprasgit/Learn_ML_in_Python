# ! /usr/bin/env python
# -*- coding: utf-8 -*-

class Trie:
    '''
    字典方法
    '''
    
    def __init__(self):
        """
        Initialize your data structure here.

        """
        self.root = {}

    def insert(self, word):
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]

        node["is_word"] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            if char in node:
                node = node[char]
            else:
                return False

        return "is_word" in node and node["is_word"]

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            if char in node:
                node = node[char]
            else:
                return False

        return True
