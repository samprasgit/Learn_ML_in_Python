

## 迷你定向网页抓取器
在调研过程中，经常需要对一些网站进行定向抓取。由于python包含各种强大的库，使用python做定向抓取比较简单。请使用python开发一个迷你定向抓取器mini_spider.py，实现对种子链接的广度优先抓取，并把URL长相符合特定pattern的网页内容（图片或者html等）保存到磁盘上。

## 程序运行

```python
python mini_spider.py -c spider.conf
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201225200000606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZmY2pqaHY=,size_16,color_FFFFFF,t_70)

## 代码架构

project

mini_spider.py: 主程序入口  
lib目录：存放不同模块的代码  
    config_load.py: 读取配置文件  
    seedfile_load.py: 读取种子文件  
    url_table.py: 构建url管理队列  
    res_table.py: 结果保存队列  
    crawl_thread.py: 实现抓取线程  
    webpage_download.py: 下载网页  
    webpage_parse.py: 对抓取网页的解析  
    webpage_save.py: 将网页保存到磁盘   
logs目录: 存放日志文件  
tests目录: 存放单测文件  
    mini_spider_test.py: 单元测试  
conf目录: 存放配置文件  
    spider.conf: 配置文件  
README.md: 说明文档  
urls: 种子文件 





参考

https://blog.csdn.net/ffcjjhv/article/details/111702024

https://github.com/DrCubic/MiniSpider