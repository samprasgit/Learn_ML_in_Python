# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Desc    :   Python爬虫获取虎扑网球员得分数据 https://nba.hupu.com/stats/players/pts/1
"""

from logging.handlers import RotatingFileHandler
import re
from urllib import response
import requests


# 定义获取HTML的方法
def get_html(page_num):
    """
    传入网页参数，获取该网页的HTML内容

    Args:
        page_num (int): 页数
    """
    url = url = "https://nba.hupu.com/stats/players/pts/" + str(page_num)
    response = requests.get(url=url)
    if response.status_code == 200:
        return response.text
    else:
        return None


# 定义正则表达式，获取所有符合正则表达式的内容
def get_data():
    # 存储数据
    point_list = []
    # 定义一个正则表达，获取球员名称，球队和得分数据
    reg_exp = '<tr>.*?<a.*?>(.*?)</a>.*?<a.*?>(.*?)</a>.*?"bg_b">(.*?)<.*?</tr>'
    for i in range(1, 6):
        html = get_html(i)
        results = re.findall(reg_exp, html, re.S)
        for result in results:
            point_list.append(result)

    return point_list


if __name__ == '__main__':
    res = get_data()
    for item in res:
        print(item)
