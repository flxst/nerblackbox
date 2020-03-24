
import logging

from os.path import abspath, dirname, join
import sys
BASE_DIR = abspath(dirname(dirname(dirname(__file__))))
sys.path.append(BASE_DIR)


class DefaultLogger:

    custom_datefmt = '%Y/%m/%d %H:%M:%S'
    custom_format = '%(asctime)s - %(name)s - %(levelname)s -  %(message)s'
    log_path = join(BASE_DIR, 'logs.log')

    @classmethod
    def clear(cls):
        """
        clear file at log_path
        """
        with open(cls.log_path, 'w') as f:
            f.write('')

    def __init__(self, path, log_path=None, level=None, mode='a'):
        """
        :param path:     [str] of python module that uses DefaultLogger, e.g. [..]/module.py
        :param log_path: [str] of log file, e.g. [..]/logs.log
        :param level:    [str] 'debug', 'info' or 'warning (minimum logging level)
        :param mode:     [str] 'w' (write) or 'a' (append)
        :created attr: logger [python logging logger]
        """

        name = path.split('/')[-1]
        self.logger = logging.getLogger(name)

        # log path
        if log_path is None:
            filename = self.log_path
        else:
            filename = join(BASE_DIR, 'logs.log')

        # level
        _level = level.upper() if level else None
        if _level:
            self.logger.setLevel(_level)

        # add file handler
        f_handler = logging.FileHandler(filename=filename, mode=mode)
        f_handler.setFormatter(logging.Formatter(self.custom_format, datefmt=self.custom_datefmt))
        if _level:
            f_handler.setLevel(_level)
        self.logger.addHandler(f_handler)

        if mode == 'w':
            self.log_info(f'ACTIVATE FILE LOGGER w/ PATH = {filename}')
        elif mode == 'a':
            self.log_info(f'APPEND TO FILE LOGGER w/ PATH = {filename}')

    def log_debug(self, *args):
        """
        debug: log & print
        """
        msg = ' '.join([str(elem) for elem in args])
        self.logger.debug(msg)     # log
        print('DEBUG -----', msg)  # print

    def log_info(self, *args):
        """
        info: log & print
        """
        msg = ' '.join([str(elem) for elem in args])
        self.logger.info(msg)      # log
        print('INFO ------', msg)  # print

    def log_warning(self, *args):
        """
        warning: log & print
        """
        msg = ' '.join([str(elem) for elem in args])
        self.logger.warning(msg)   # log
        print('WARNING ---', msg)  # print