# !/usr/bin/python
# -*- coding: utf-8 -*-


class URL:
    def __init__(self, url, depth=0):
        self.__url = url
        self.__depth = depth

    def get_url(self):
        return self.__url

    def get_depth(self):
        return self.__depth
