Python面向对象编程

## 类和对象

>  **对象：用来描述客观事物的一个实体，由一组属性和方法构成**

对象同时具有属性和方法两项特性。 对象的属性和方法通常被封装在一起，共同体现事物的特性， 二者相辅相承，不能分割。

> **类是模子，定义对象将会拥有的特征（属性）和行为（方法）**

> **根据类来创建对象被称为实例化**

## 类的定义

### 只包含方法的类

```python
class 类名:

    def 方法1(self, 参数列表):
        pass

    def 方法2(self, 参数列表):
        pass
```

```python
class Cat:
    """这是一个猫类"""

    def eat(self):
        print("小猫爱吃鱼")

    def drink(self):
        print("小猫在喝水")

tom = Cat()
tom.drink()
tom.eat()
```

### 创建对象

```undefined
对象变量 = 类名()
```

### Self参数

给对象增加参数

**init** 方法是 专门 用来定义一个类 具有哪些属性并且给出这些属性的初始值的方法！









## 参考：

https://www.jb51.net/article/160159.htm

https://www.jb51.net/article/159985.htm