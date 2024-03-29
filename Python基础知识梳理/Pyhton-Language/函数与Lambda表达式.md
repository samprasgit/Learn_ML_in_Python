<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [函数与Lambda表达式](#%E5%87%BD%E6%95%B0%E4%B8%8Elambda%E8%A1%A8%E8%BE%BE%E5%BC%8F)
  - [1. 函数](#1-%E5%87%BD%E6%95%B0)
    - [函数的定义](#%E5%87%BD%E6%95%B0%E7%9A%84%E5%AE%9A%E4%B9%89)
    - [函数的调用](#%E5%87%BD%E6%95%B0%E7%9A%84%E8%B0%83%E7%94%A8)
    - [函数文档](#%E5%87%BD%E6%95%B0%E6%96%87%E6%A1%A3)
    - [函数参数](#%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0)
    - [函数的返回值](#%E5%87%BD%E6%95%B0%E7%9A%84%E8%BF%94%E5%9B%9E%E5%80%BC)
    - [变量作用域](#%E5%8F%98%E9%87%8F%E4%BD%9C%E7%94%A8%E5%9F%9F)
  - [2. Lambda 表达式](#2-lambda-%E8%A1%A8%E8%BE%BE%E5%BC%8F)
    - [匿名函数的定义](#%E5%8C%BF%E5%90%8D%E5%87%BD%E6%95%B0%E7%9A%84%E5%AE%9A%E4%B9%89)
    - [匿名函数的应用](#%E5%8C%BF%E5%90%8D%E5%87%BD%E6%95%B0%E7%9A%84%E5%BA%94%E7%94%A8)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 函数与Lambda表达式

## 1. 函数

还记得 Python 里面“万物皆对象”么？Python 把函数也当成对象，可以从另一个函数中返回出来而去构建高阶函数，比如：
- 参数是函数 
- 返回值是函数


### 函数的定义
- 函数以`def`关键词开头，后接函数名和圆括号()。
- 函数执行的代码以冒号起始，并且缩进。
- return [表达式] 结束函数，选择性地返回一个值给调用方。不带表达式的return相当于返回`None`。


```python
def functionname(parameters):
    "函数_文档字符串"
    function_suite
    return [expression]
```


### 函数的调用

【例子】

```python
def printme(str):
    print(str)


printme("我要调用用户自定义函数!")  # 我要调用用户自定义函数!
printme("再次调用同一函数")  # 再次调用同一函数
temp = printme('hello') # hello
print(temp)  # None
```

【例子】
```python
def add(a, b):
    print(a + b)


add(1, 2)  # 3
add([1, 2, 3], [4, 5, 6]) # [1, 2, 3, 4, 5, 6]
```

### 函数文档

```python
def MyFirstFunction(name):
    "函数定义过程中name是形参"
    # 因为Ta只是一个形式，表示占据一个参数位置
    print('传递进来的{0}叫做实参，因为Ta是具体的参数值！'.format(name))


MyFirstFunction('老马的程序人生')  
# 传递进来的老马的程序人生叫做实参，因为Ta是具体的参数值！

print(MyFirstFunction.__doc__) # 输出函数注释 
# 函数定义过程中name是形参

help(MyFirstFunction)
# Help on function MyFirstFunction in module __main__:
# MyFirstFunction(name)
#    函数定义过程中name是形参
```

### 函数参数

Python 的函数具有非常灵活多样的参数形态，既可以实现简单的调用，又可以传入非常复杂的参数。从简到繁的参数形态如下：
- 位置参数 (positional argument)
- 默认参数 (default argument)
- 可变参数 (variable argument)
- 关键字参数 (keyword argument)
- 命名关键字参数 (name keyword argument)
- 参数组合


**1. 位置参数**
```python
def functionname(arg1):
    "函数_文档字符串"
    function_suite
    return [expression]
```
- `arg1` - 位置参数 ，这些参数在调用函数 (call function) 时位置要固定。

**2. 默认参数**

```python
def functionname(arg1, arg2=v):
    "函数_文档字符串"
    function_suite
    return [expression]
```
- `arg2 = v` - 默认参数 = 默认值，调用函数时，默认参数的值如果没有传入，则被认为是默认值。
- 默认参数一定要放在位置参数 <b>后面</b>，不然程序会报错。

【例子】
```python
def printinfo(name, age=8):
    print('Name:{0},Age:{1}'.format(name, age))


printinfo('小马')  # Name:小马,Age:8
printinfo('小马', 10)  # Name:小马,Age:10
```

- Python 允许函数调用时参数的顺序与声明时不一致，因为 Python 解释器能够用参数名匹配参数值。

【例子】
```python
def printinfo(name, age):
    print('Name:{0},Age:{1}'.format(name, age))


printinfo(age=8, name='小马')  # Name:小马,Age:8
```


**3. 可变参数**

顾名思义，可变参数就是传入的参数个数是可变的，可以是 0, 1, 2 到任意个，是不定长的参数。

```python
def functionname(arg1, arg2=v, *args):
    "函数_文档字符串"
    function_suite
    return [expression]
```
- `*args` - 可变参数，可以是从零个到任意个，自动组装成元组。
- 加了星号（*）的变量名会存放所有未命名的变量参数。

【例子】
```python
def printinfo(arg1, *args):
    print(arg1)
    for var in args:
        print(var)


printinfo(10)  # 10
printinfo(70, 60, 50)

# 70
# 60
# 50
```



**4. 关键字参数**

```python
def functionname(arg1, arg2=v, *args, **kw):
    "函数_文档字符串"
    function_suite
    return [expression]
```
- `**kw` - 关键字参数，可以是从零个到任意个，自动组装成字典。

【例子】
```python
def printinfo(arg1, *args, **kwargs):
    print(arg1)
    print(args)
    print(kwargs)


printinfo(70, 60, 50)
# 70
# (60, 50)
# {}
printinfo(70, 60, 50, a=1, b=2)
# 70
# (60, 50)
# {'a': 1, 'b': 2}
```

「可变参数」和「关键字参数」的同异总结如下：
- 可变参数允许传入零个到任意个参数，它们在函数调用时自动组装为一个元组 (tuple)。
- 关键字参数允许传入零个到任意个参数，它们在函数内部自动组装为一个字典 (dict)。

**5. 命名关键字参数**
```python
def functionname(arg1, arg2=v, *args, *, nkw, **kw):
    "函数_文档字符串"
    function_suite
    return [expression]
```
- `*, nkw` - 命名关键字参数，用户想要输入的关键字参数，定义方式是在nkw 前面加个分隔符 *。
- 如果要限制关键字参数的名字，就可以用「命名关键字参数」
- 使用命名关键字参数时，要特别注意不能缺少参数名。

【例子】
```python
def printinfo(arg1, *, nkw, **kwargs):
    print(arg1)
    print(nkw)
    print(kwargs)


printinfo(70, nkw=10, a=1, b=2)
# 70
# 10
# {'a': 1, 'b': 2}

printinfo(70, 10, a=1, b=2)
# TypeError: printinfo() takes 1 positional argument but 2 were given
```
- 没有写参数名`nwk`，因此 10 被当成「位置参数」，而原函数只有 1 个位置函数，现在调用了 2 个，因此程序会报错。


**6. 参数组合**

在 Python 中定义函数，可以用位置参数、默认参数、可变参数、命名关键字参数和关键字参数，这 5 种参数中的 4 个都可以一起使用，但是注意，参数定义的顺序必须是：
- 位置参数、默认参数、可变参数和关键字参数。
- 位置参数、默认参数、命名关键字参数和关键字参数。

要注意定义可变参数和关键字参数的语法：
- `*args` 是可变参数，`args` 接收的是一个 `tuple`
- `**kw` 是关键字参数，`kw` 接收的是一个 `dict`

命名关键字参数是为了限制调用者可以传入的参数名，同时可以提供默认值。定义命名关键字参数不要忘了写分隔符 `*`，否则定义的是位置参数。

警告：虽然可以组合多达 5 种参数，但不要同时使用太多的组合，否则函数很难懂。


### 函数的返回值

【例子】

```python
def add(a, b):
    return a + b


print(add(1, 2))  # 3
print(add([1, 2, 3], [4, 5, 6]))  # [1, 2, 3, 4, 5, 6]
```

【例子】

```python
def back():
    return [1, '小马的程序人生', 3.14]


print(back())  # [1, '小马的程序人生', 3.14]
```

【例子】

```python
def back():
    return 1, '小马的程序人生', 3.14


print(back())  # (1, '小马的程序人生', 3.14)
```

【例子】
```python
def printme(str):
    print(str)

temp = printme('hello') # hello
print(temp) # None
print(type(temp))  # <class 'NoneType'>
```

### 变量作用域

- Python 中，程序的变量并不是在哪个位置都可以访问的，访问权限决定于这个变量是在哪里赋值的。
- 定义在函数内部的变量拥有局部作用域，该变量称为局部变量。
- 定义在函数外部的变量拥有全局作用域，该变量称为全局变量。
- 局部变量只能在其被声明的函数内部访问，而全局变量可以在整个程序范围内访问。

【例子】
```python
def discounts(price, rate):
    final_price = price * rate
    return final_price


old_price = float(input('请输入原价:'))  # 98
rate = float(input('请输入折扣率:'))  # 0.9
new_price = discounts(old_price, rate)
print('打折后价格是:%.2f' % new_price)  # 88.20
```
- 当内部作用域想修改外部作用域的变量时，就要用到`global`和`nonlocal`关键字了。

【例子】
```python
num = 1


def fun1():
    global num  # 需要使用 global 关键字声明
    print(num)  # 1
    num = 123
    print(num)  # 123


fun1()
print(num)  # 123
```

**内嵌函数**

【例子】
```python
def outer():
    print('outer函数在这被调用')

    def inner():
        print('inner函数在这被调用')

    inner()  # 该函数只能在outer函数内部被调用


outer()
# outer函数在这被调用
# inner函数在这被调用
```

**闭包**

- 是函数式编程的一个重要的语法结构，是一种特殊的内嵌函数。
- 如果在一个内部函数里对外层非全局作用域的变量进行引用，那么内部函数就被认为是闭包。
- 通过闭包可以访问外层非全局作用域的变量，这个作用域称为 <b>闭包作用域</b>。

【例子】

```python
def funX(x):
    def funY(y):
        return x * y

    return funY


i = funX(8)
print(type(i))  # <class 'function'>
print(i(5))  # 40
```

【例子】闭包的返回值通常是函数。

```python
def make_counter(init):
    counter = [init]

    def inc(): counter[0] += 1

    def dec(): counter[0] -= 1

    def get(): return counter[0]

    def reset(): counter[0] = init

    return inc, dec, get, reset


inc, dec, get, reset = make_counter(0)
inc()
inc()
inc()
print(get())  # 3
dec()
print(get())  # 2
reset()
print(get())  # 0
```

【例子】 如果要修改闭包作用域中的变量则需要 `nonlocal` 关键字
```python
def outer():
    num = 10

    def inner():
        nonlocal num  # nonlocal关键字声明
        num = 100
        print(num)

    inner()
    print(num)


outer()

# 100
# 100
```

**递归**

- 如果一个函数在内部调用自身本身，这个函数就是递归函数。

【例子】`n! = 1 x 2 x 3 x ... x n`

++循环++

```python
n = 5
for k in range(1, 5):
    n = n * k
print(n)  # 120
```


++递归++
```python
def factorial(n):
    if n == 1:
        return 1
    return n * fact(n - 1)


print(factorial(5)) # 120
```

【例子】斐波那契数列 `f(n)=f(n-1)+f(n-2), f(0)=0 f(1)=1`

++循环++

```python
i = 0
j = 1
lst = list([i, j])
for k in range(2, 11):
    k = i + j
    lst.append(k)
    i = j
    j = k
print(lst)  
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

++递归++

```python
def recur_fibo(n):
    if n <= 1:
        return n
    return recur_fibo(n - 1) + recur_fibo(n - 2)


lst = list()
for k in range(11):
    lst.append(recur_fibo(k))
print(lst)  
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```


【例子】设置递归的层数，Python默认递归层数为 100
```python
import sys

sys.setrecursionlimit(1000)
```

---
## 2. Lambda 表达式

### 匿名函数的定义

在 Python 里有两类函数：
- 第一类：用 `def` 关键词定义的正规函数
- 第二类：用 `lambda` 关键词定义的匿名函数

python 使用 `lambda` 关键词来创建匿名函数，而非`def`关键词，它没有函数名，其语法结构如下：

```python
lambda argument_list: expression
```
- `lambda` - 定义匿名函数的关键词。
- `argument_list` - 函数参数，它们可以是位置参数、默认参数、关键字参数，和正规函数里的参数类型一样。
- `:`- 冒号，在函数参数和表达式中间要加个冒号。
- `expression` - 只是一个表达式，输入函数参数，输出一些值。

注意：
- `expression` 中没有 return 语句，因为 lambda 不需要它来返回，表达式本身结果就是返回值。
- 匿名函数拥有自己的命名空间，且不能访问自己参数列表之外或全局命名空间里的参数。

【例子】
```python
def sqr(x):
    return x ** 2


print(sqr)
# <function sqr at 0x000000BABD3A4400>

y = [sqr(x) for x in range(10)]
print(y)
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

lbd_sqr = lambda x: x ** 2
print(lbd_sqr)
# <function <lambda> at 0x000000BABB6AC1E0>

y = [lbd_sqr(x) for x in range(10)]
print(y)
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


sumary = lambda arg1, arg2: arg1 + arg2
print(sumary(10, 20))  # 30

func = lambda *args: sum(args)
print(func(1, 2, 3, 4, 5))  # 15
```


### 匿名函数的应用

函数式编程 是指代码中每一块都是不可变的，都由纯函数的形式组成。这里的纯函数，是指函数本身相互独立、互不影响，对于相同的输入，总会有相同的输出，没有任何副作用。

【例子】非函数式编程
```python
def f(x):
    for i in range(0, len(x)):
        x[i] += 10
    return x


x = [1, 2, 3]
f(x)
print(x)
# [11, 12, 13]
```

【例子】函数式编程
```python
def f(x):
    y = []
    for item in x:
        y.append(item + 10)
    return y


x = [1, 2, 3]
f(x)
print(x)
# [1, 2, 3]
```

匿名函数 常常应用于函数式编程的高阶函数 (high-order function)中，主要有两种形式：
- 参数是函数 (filter, map)
- 返回值是函数 (closure)


如，在 `filter`和`map`函数中的应用：

- `filter(function, iterable)` 过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 `list()` 来转换。

【例子】
```python
odd = lambda x: x % 2 == 1
templist = filter(odd, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(list(templist))  # [1, 3, 5, 7, 9]
```

- `map(function, *iterables)` 根据提供的函数对指定序列做映射。

【例子】
```python
m1 = map(lambda x: x ** 2, [1, 2, 3, 4, 5])
print(list(m1))  
# [1, 4, 9, 16, 25]

m2 = map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
print(list(m2))  
# [3, 7, 11, 15, 19]
```



除了 Python 这些内置函数，我们也可以自己定义高阶函数。

【例子】
```python
def apply_to_list(fun, some_list):
    return fun(some_list)

lst = [1, 2, 3, 4, 5]
print(apply_to_list(sum, lst))
# 15

print(apply_to_list(len, lst))
# 5

print(apply_to_list(lambda x: sum(x) / len(x), lst))
# 3.0
```

---
**参考文献**：
- https://www.runoob.com/python3/python3-tutorial.html
- https://www.bilibili.com/video/av4050443
- https://mp.weixin.qq.com/s/gKhXS8JVU8dZBHJF7sIFsw

---
