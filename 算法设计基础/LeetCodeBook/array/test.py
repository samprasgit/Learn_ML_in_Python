# ! /usr/bin/env python
# -*- coding: utf-8 -*-

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
