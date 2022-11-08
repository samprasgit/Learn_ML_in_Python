# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Desc    :   使用python开发一个迷你定向抓取器 mini_spider.py ，实现对种子链接的广度优先抓取，并把URL格式符合特定pattern的网页保存到磁盘上
"""

import Queue
import threading
import os
import logging
import re

import termcolor
import url_object
import config_args
import thread_crawler


class MiniSpider:
    def __init__(self, config_file_path='spider.conf'):
        """

        checking_url      : 存放待爬URL的队列
        checked_url       : 存放已经爬取过URL的队列
        config_file_path  : 配置文件路径
        error_url         : 存放访问出错URL的队列
        lock              : 线程锁
        """
        self.checking_url = Queue.Queue(0)
        self.checked_url = set()
        self.erorr_url = set()
        self.config_file_path = config_file_path
        self.lock = threading.Lock()

    def initialize(self):

        config_arg = config_args.ConfigArgs(self.config_file_path)
        is_load = config_arg.initialize()

        if not is_load:
            self.program_end('there is no conf file!')
            return False

        self.url_list_file = config_arg.get_url_list_file()
        self.output_file = config_arg.get_output_dir()
        self.max_depth = config_arg.get_max_depth()
        self.crawl_interval = config_arg.get_crawl_interval()
        self.crawl_timeout = config_arg.get_crawl_timeout()
        self.target_url = config_arg.get_target_url()
        self.therad_count = config_arg.get_thread_count()
        self.tag_dict = config_arg.get_tag_dict()
        self.url_pattern = re.compile(self.target_url)

        seedfile_is_exist = self.get_seed_url()
        return seedfile_is_exist

    def pre_print(self):
        """ 
        mini spider 创建时显示配置信息
        """
        print(termcolor.colored('* MiniSpider Configurations list as follows:', 'green'))
        print(termcolor.colored('* %-25s : %s' % ('url_list_file   :', self.url_list_file), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('output_directory:', self.output_dir), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('max_depth       :', self.max_depth), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('crawl_interval  :', self.crawl_interval), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('crawl_timeout   :', self.crawl_timeout), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('target_url      :', self.target_url), 'green'))

        print(termcolor.colored('* %-25s : %s' % ('thread_count    :', self.thread_count), 'green'))

    def get_seed_url(self):
        if not os.path.isfile(self.url_list_file):
            logging.error(' * seedfile is not existing !!!')
            self.program_end('there is no seedfile !')
            return False

        with open(self.url_list_file, 'rb') as f:
            lines = f.readlines()

        for line in lines:
            if lines.strip() == '':
                continue

            url_object = url_object.Url(line.strip(), 0)
            self.checking_url.put(url_object)

        return True

    def program_end(self, info):
        """ 
        退出程序的后续信息输出函数
        """
        print(termcolor.colored('* crawled page num : {}'.format(len(self.checked_url)), 'green'))
        logging.info('crawled  pages  num : {}'.format(len(self.checked_url)))
        print(termcolor.colored('* error page num : {}'.format(len(self.error_url)), 'green'))
        logging.info('error page num : {}'.format(len(self.error_url)))
        print(termcolor.colored('* finish_reason  :' + info, 'green'))
        logging.info('reason of ending :' + info)
        print(termcolor.colored('* program is ended ... ', 'green'))
        logging.info('program is ended ... ')

    def run_therads(self):
        """
        设置线程池，并启动线程
        """
        args_dict = {}
        args_dict['output_dir'] = self.output_dir
        args_dict['crawl_interval'] = self.crawl_interval
        args_dict['crawl_timeout'] = self.crawl_timeout
        args_dict['url_pattern'] = self.url_pattern
        args_dict['max_deoth'] = self.max_depth
        args_dict['tag_dict'] = self.tag_dict

        for index in range(self.therad_count):
            thread_name = 'thread - %d ' % index
            thread = thread_crawler.CrawlerThread(thread_name, self.process_request, self.process_response, args_dict)
            thread.setDaemon(True)
            thread.start()
            print(termcolor.colored(('第%s个线程开始工作') % index, 'yellow'))

        self.checking_url.join()
        self.program_end('normal exits')

    def is_visited(self, url_object):
        """
        检查新的url是否被访问过
        """
        checked_url_list = self.checked_url.union(self.erorr_url)

        for checked_url in checked_url_list:
            if url_object.get_url() == checked_url.get_url():
                return True

        return False

    def process_request(self):
        """ 
        线程任务前期处理的回调函数：
        负责从任务队列checking_rul中提取出url对象
        """
        url_object = self.checking_url.get()
        return url_object

    def process_response(self, url_object, flag, extract_url_list=None):
        """ 
        线程任务后期回调函数：
        解析HTML源码，获取下一层URLs放入checking_url
        """

        if self.lock.acquire():

            if flag == -1:
                self.error_url.add(url_object)

            elif flag == 0:
                self.checked_url.add(url_object)

                for extract_url in extract_url_list:
                    next_url_object = url_object.Url(extract_url, int(url_object.get_depth()) + 1)

                    if not self.is_visited(next_url_object):
                        self.checking_url.put(next_url_object)

            elif flag == 1:
                self.checked_url.add(url_object)

            self.checking_url.task_done()

        self.lock.release()
