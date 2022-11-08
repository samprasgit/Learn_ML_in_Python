# !/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import socket
import urllib


class HtmlDownloader:
    """
    下载url_object对应的html源码

    """
    def __init__(self, url_object, timeout, try_times):
        """

        :param  url_object: url对象
        :param  timeout: 下载时间延迟
        :param  try_times: 下载尝试次数
        """

        self.url_object = url_object
        self.timeout = timeout
        self.try_times = try_times

    def download(self):
        """ 
        检查url_object是否可以被访问，若能访问则放入checked_url,否则放入error_url
        
        Response:
                None : URL访问成功与失败
                0/-1 : 访问成功与失败
        """

        # 开始下载
        for t in range(self.try_times):
            try:
                response = urllib.request.urlopen(self.url_object.get_ulr(), timeout=self.timeout)
                response.depth = self.url_object.get_depth()
                return (response, 0)

            except urllib.error.URLError as e:
                if t == self.try_times - 1:
                    error_info = '*Downloading failed : %s - %s' % (self.url_object.get_url(), e)

            except UnicodeDecodeError as e:
                if t == self.try_times - 1:
                    error_info = '*Downloading failed : %s - %s' % (self.url_object.get_url(), e)

            except urllib.error.HTTPError as e:
                error_info = '*Downloading failed : %s - %s' % (self.url_object.get_url(), e)

            except socket.timeout as e:
                if t == self.try_times - 1:
                    error_info = '*Downloading failed : %s - %s' % (self.url_object.get_url(), e)

            except Exception as e:
                if t == self.try_times - 1:
                    error_info = '*Downloading failed : %s - %s' % (self.url_object.get_url(), e)

            logging.warn(' * Try for ()th times'.format(t + 1))

            if t == self.try_times - 1:
                logging.warn(error_info)
                return (None, -1)


if __name__ == "__main__":
    url = 'www.baidu.com'
    timeout = 1
    try_times = 3
    html_downloader = HtmlDownloader(url, timeout, try_times)
    print(html_downloader)
