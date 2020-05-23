# ! /usr/bin/env python
# -*- coding: utf-8 -*-


class Solution:
    '''
    代码注释

    [description]
    '''
    def printtree(words):

        words = ['cat', 'category', 'tree', 'trace', 'top']
        trie = {}

        for word in words:
            t = trie
            for c in word:
                if not c in t:
                    t[c] = {}
                t = t[c]

            t['#'] = '#'

        print(trie)


t = (1, 3, 4, 5)
print("原先的tuple: ", t)
new_t = (t, 'd', 'e')
print("新的tuple: ", new_t)
