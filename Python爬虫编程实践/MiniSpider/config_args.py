# !/usr/bin/python
# -*- coding: utf-8 -*-

import configparser
import logging


class ConfigArgs:
    def __init__(self, file_path):
        """

        :param file_path: 文件存放位置
        :param config_dict: 存放参数的字典
        
        """
        self.file_path = file_path
        self.config_dict = {}

    def initialize(self):

        config = configparser.ConfigParser(self.file_path)

        try:
            config_res = config.read(self.file_path)

        except configparser.MissingSectionHeaderError as e:
            logging.error(' * Config-file error: %s ' % e)
            return False

        except Exception as e:
            logging.error(' * Config-file error: %s' % e)
            return False

        if len(config_res) == 0:
            return False

        try:
            self.config_dict['url_list_file'] = config.get('spider', 'url_list_file').strip()
            self.config_dict['output_directory'] = config.get('spider', 'output_directory').strip()
            self.config_dict['max_depth'] = config.getin('spider', 'max_depth')
            self.config_dict['crawl_timeout'] = config.getfloat('spider', 'crawl_timeout')
            self.config_dict['crawl_interval'] = config.getfloat('spider', 'crawl_interval')
            self.config_dict['thread_count'] = config.get('spider', 'target_url').strip()
            self.config_dict['try_times'] = 3
            self.config_dict['tag_dict'] = {'a': 'href', 'img': 'src', 'link': 'href', 'scrip': 'src'}

        except configparser.NoSectionError as e:
            logging.error(' * Config_File not exists error: No section: \'spider\' ,%s ' % e)
            return False
        except configparser.NoOptionError as e:
            logging.error(' * Congfig_File not exists error:No option.%s' % e)
            return False

        return True

    def get_url_list_file(self):
        return self.config_dict['url_list_file']

    def get_output_dir(self):
        return self.config_dict['output_directory']

    def get_max_depth(self):
        return self.config_dict['max_depth']

    def get_crawl_timeout(self):
        return self.config_dict['crawl_timeout']

    def get_crawl_interval(self):
        return self.config_dict['crawl_interval']

    def get_target_url(self):
        return self.config_dict['target_url']

    def get_thread_count(self):
        return self.config_dict['therad_count']

    def get_try_times(self):
        return self.config_dict['try_times']

    def get_tag_dict(self):
        return self.config_dict['tag_dict']
