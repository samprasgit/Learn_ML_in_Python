# !/usr/bin/python
# -*- coding: utf-8 -*-

import urllib.request
import socket
import logging


class HtmlDownloader:
    """
    下载url_object对应的html源码
    :param  url_object: url对象
    :param  timeout: 下载时间延迟
    :param  try_times: 下载尝试次数
    """

         
    def __init__(self, url_object, timeout, try_times):
        
        self.url_object = url_object
        self.timeout = timeout
        self.try_times = try_times
        