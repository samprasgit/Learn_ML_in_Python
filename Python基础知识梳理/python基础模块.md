<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [pprint](#pprint)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## pprint   

print()和pprint()都是python的打印模块，功能基本一样，唯一的区别就是pprint()模块打印出来的数据结构更加完整，每行为一个数据结构，更加方便阅读打印输出结果。特别是对于特别长的数据打印，print()输出结果都在一行，不方便查看，而pprint()采用分行打印输出，所以对于数据结构比较复杂、数据长度较长的数据，适合采用pprint()打印方式。当然，一般情况多数采用print()。

```python
import pprint

data = ("test", [1, 2, 3, 'test', 4, 5], "This is a string!",
        {'age': 23, 'gender': 'F'})
print(data)
pprint.pprint(data)


输出结果
('test', [1, 2, 3, 'test', 4, 5], 'This is a string!', {'age': 23, 'gender': 'F'})
('test',
 [1, 2, 3, 'test', 4, 5],
 'This is a string!',
 {'age': 23, 'gender': 'F'})

```

collections

