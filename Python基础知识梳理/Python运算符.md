- [算术运算符](https://www.runoob.com/python/python-operators.html#ysf1)
- [比较（关系）运算符](https://www.runoob.com/python/python-operators.html#ysf2)
- [赋值运算符](https://www.runoob.com/python/python-operators.html#ysf3)
- [逻辑运算符](https://www.runoob.com/python/python-operators.html#ysf4)
- [位运算符](https://www.runoob.com/python/python-operators.html#ysf5)
- [成员运算符](https://www.runoob.com/python/python-operators.html#ysf6)
- [身份运算符](https://www.runoob.com/python/python-operators.html#ysf7)
- [运算符优先级](https://www.runoob.com/python/python-operators.html#ysf8)





### 1.混淆点

>- is, is not 对比的是两个变量的内存地址
>- ==, != 对比的是两个变量的值
>- 比较的两个变量，指向的都是地址不可变的类型（str等），那么is，is not 和 ==，！= 是完全等价的。
>- 对比的两个变量，指向的是地址可变的类型（list，dict，tuple等），则两者是有区别的。

### 2.运算符优先级

| 运算符            | 描述                     |
| ----------------- | ------------------------ |
| **                | 指数（最高优先级）       |
| ~+-               | 按位翻转，一元加号和减号 |
| * / % //          | 乘，除，取模和取整除）   |
| + -               | 加法减法                 |
| >> <<             | 右移，左移运算符         |
| &                 | 位‘AND’                  |
| ^\|               | 位运算符                 |
| <=<>>=            | 比较运算符               |
| <>==!=            | 等于运算符               |
| =%=/=//=-=+=*=**= | 赋值运算符               |
| is is not         | 身份运算符               |
| in not in         | 成员运算符               |
| not and or        | 逻辑运算符               |

