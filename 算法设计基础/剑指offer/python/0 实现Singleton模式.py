#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/2/9 6:06 下午
# Author : samprasgit
# desc :  单例模式

"""
方法1：实现__new__方法，然后将类的一个实例绑定到类变量_instance上
如果cls._instance为None，说明该类没有被实例化过，new一个该类的实例，并返回
如果cls._instance不是None，直接返回——instance
"""


class Singleton1:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton1, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)

        return cls._instance


"""
方法2：共享属性：所谓单例就是所有引用（实例、对象）拥有相同的状态（属性）和行为（方法）
同一个类的所有实例天然有相同的行为（方法）
只需要保证一个类的所有实例具有相同的状态（属性）即可
所有实例共享属性的最简单方法就是__dict__属性指向（引用）同一个字典（dict）
"""


class Singleton2:
    _state = {}

    def __new__(cls, *args, **kwargs):
        ob = super(Singleton2, self).__new__(cls, *args, **kwargs)
        ob.__dict__ = cls._state
        return ob


"""
方法3：装饰器版本decorator
这是一种更pythonic,更elegant的方法
单例类本身根本不知道自己是单例的，因为他自己的代码不是单例的
"""


def singleton(cls, *args, **kwargs):
    instance = {}

    def getinstance():
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)

        return instance[cls]

    return getinstance


@singleton
class MyClass:
    a = 1

    def __init__(self, x=0):
        self.x = 0
