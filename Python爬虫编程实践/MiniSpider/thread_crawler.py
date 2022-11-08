# !/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import time
import threading
import urllib
import re

import html_download
import html_parser


class CrawlerThread(therading.Thread):
    def __init__(self, name, process_request, process_response, args_dict):
        super(CrawlerThread, self).__init__(name=name)
        self.process_request = process_request
        self.process_response = process_response
        self.output_dir = args_dict['output_dir']
        self.crawl_interval = args_dict['crawl_interval']
        self.crawl_timeout = args_dict['crawl_timeout']
        self.url_pattern = args_dict['url_pattern']
        self.max_depth = args_dict['max_depth']
        self.tag_dict = args_dict['tag_dict']

    def run(self):
        """ 
        线程执行的具体内容
        """
        while True:
            url_object = self.process_request()
            time.sleep(self.crawl_interval)

            logging.inof('%-12s : get a url in depth : ' % threading.currentThread.getName() + str(url_object.get_depth()))

            if self.is_target_url(url_object.get_url()):
                flag = -1
                if self.save_target(url_object.get_url()):
                    flag = 1

                self.process_response(url_object, flag)
                continue

            if url_object.get_depth() < self.max_depth:
                downloader_object = html_download.HtmlDownloader(url_object, self.crawl_timeout)
                response, flag = downloader_object.download()

                if flag == -1:
                    self.process_response(url_object, flag)
                    continue

                if flag == 0:
                    content = response.read()
                    url = url_object.get()
                    soup = html_parser.HtmlParse(content, self.tag_dict, url)
                    extract_url_list = soup.extract_url()

                    self.process_response(url_object, flag, extract_url_list)

            else:
                flag = 2  # depth > max_depth的正常url
                self.process_response(url_object, flag)

    def is_target_url(self, url):
        """ 
        判断url是否符合target url的形式
        """
        found_aim = self.url_pattern.match(url)
        if found_aim:
            return True
        return False

    def save_target(self, url):

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        file_name = urllib.parse.quote_plus(url)

        if len(file_name) > 127:
            file_name = file_name[-127:]

        target_path = "{}/{}".format(self.output_dir, file_name)

        try:
            urllib.request.urlretrieve(url, target_path)
            return True
        except IOError as e:
            logging.warn(' * Save target failed:%s - %s' (url, e))
            return False
