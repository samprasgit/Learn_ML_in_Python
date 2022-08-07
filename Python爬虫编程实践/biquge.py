# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time    :   2022/08/06 21:22:11
@Desc    :   使用Python获取笔趣阁中辰东最新小说
"""
from email import header
import requests
import os
import re
import time
from bs4 import BeautifulSoup

url = "https://www.xbiquge.so"  # 笔趣阁小说网站地址
url1 = 'https://www.xbiquge.la/82/82620/' # 辰东新作《星空彼岸》网址
test_url="https://www.xbiquge.so/book/43106/37044337.html"
# --------------------获取小说页详情---------------------
def get_html(url):

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        result = response.text  
        return result  

    except:
        
        return "Error"


# print(get_html(url1))


# -----------爬取当前页面所有的章节的url-----------------------
def get_url_all(url):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"}
    try:
        # 打开网页
        response=requests.get(url,headers=headers)
        response.raise_for_status() 
        response.encoding=response.apparent_encoding 
        response=str(response.text)
        # 创建BeautifulSoup对象，用于解析网页获取所需内容
        url_1=BeautifulSoup(response,'html.parser')
        # 找到指定的标签列表
        # 每个章节对应的url存放在dd标签下a标签中的href属性下
        result= url_1.find_all('dd')
        url_list= []
        for i in range(len(result)):
            url_list.append(re.findall(r"href=.*html",str(result[i])))            
        return url_list
 
    except:
        return "Error"
    
    
html_1=get_url_all(test_url)


# 获取小说正文
def find_text(html):
    texts=re.findall(r'id="content"(.*)',html)
    texts=str(texts)
    texts=re.split(r"\\r<br />\\r<br />&nbsp;&nbsp;&nbsp;&nbsp;",texts)  
    res = " " 
    for i in range(len(texts)):
        res+= " "+texts[i]+"\n"
        
    return res 


# print(find_text(html_1)) 


# 爬取小说全文
book_url= 'https://www.xbiquge.la/82/82620/' # 辰东新作《星空彼岸》网址
def get_book(url):
    r= open("./辰东小说.txt",'w+',encoding='utf-8')
    url_list=get_url_all(book_url)
    for i in range(10):
        print("开始爬去小说第 ",i," 章")
        url=str(url_list[i])
        url = "https://www.xbiquge.la" + url[8:-2]
        # 获取小说当前章节的html 
        html=get_html(url)
        text=find_text(html)
        r.write(text)
    r.close()  
    print("爬取成功！")
    
print(get_book(book_url))