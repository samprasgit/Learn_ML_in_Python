# !/usr/bin/python
# -*- coding: utf-8 -*-

import logging
from urllib.parse import urlparse

import bs4
import chardet


class HtmlParse:
    def __init__(self, content, link_tag_dict, url):
        """_summary_

        :param  content: 待解析的html源码
        :param  link_tag_dict: 待解析的标签
        :param  url: 带解析页面的url
        """
        self.link_tag_dict = link_tag_dict
        self.content = content
        self.url = url

    def extract_url(self):
        extract_url_list = []
        if not self.enc_to_utf8():
            return extract_url_list

        host_name = urlparse(self.url).netloc

        soup = bs4.BeautifulSoup(self.content, 'html5lib')

        for tag, attr in self.link_tag_dict.iteritems():
            all_found_tags = soup.find_all(tag)
            for found_tag in all_found_tags:
                extract_url = found_tag.get(attr).strip()

                if extract_url.startswith('javascript') or len(extract_url) > 256:
                    continue

                if not (extract_url.startswith('http:') or extract_url.startswith('https:')):
                    extract_url = urlparse.urljoin(self.url, extract_url)

                extract_url_list.append(extract_url)

        return extract_url_list

    def detect_encoding(self):
        """
        检测html文本编码
        """

        if isinstance(self.content, unicode):
            return 'unicode'

        try:
            encode_dict = chardet.detect(self.content)
            encode_name = encode_dict['encoding']
            return encode_name
        except Exception as e:
            logging.error(' * Error coding-detect: %s ' % e)
            return None

    def enc_to_utf8(self):
        """ 
        将文本转为utf8
        """
        encoding = self.detect_encoding()
        try:
            if encoding is None:
                return False

            elif encoding.lower() == 'unicode':
                self.content = self.content.encode('utf-8')
                return True

            elif encoding.lower() == 'utf-8':
                return True

            else:
                self.content = self.content.decode(encoding, 'ignore').encode('utf-8')

        except UnicodeError as e:
            logging.warn(' * EncodingError - %s - %s ' % (self.url, e))
            return False

        except UnicodeEncodeError as e:
            logging.warn(' * EncodingError - %s - %s ' % (self.url, e))
            return False

        except UnicodeDecodeError as e:
            logging.warn(' * EncodingError - %s - %s ' % (self.url, e))
            return False
        except Exception as e:
            logging.warn(' * EncodingError - %s - %s ' % (self.url, e))
            return False
