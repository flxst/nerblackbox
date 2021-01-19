import logging

import os
from os.path import join

BASE_DIR = os.environ.get("BASE_DIR")


class DefaultLogger:

    custom_datefmt = "%Y/%m/%d %H:%M:%S"
    custom_format = "%(asctime)s - %(name)s - %(levelname)s -  %(message)s"
    assert BASE_DIR is not None, "ERROR! BASE_DIR not found."
    default_log_file = join(BASE_DIR, "logs.log")

    def __init__(self, path, log_file=None, level=None, mode="a"):
        """
        :param path:     [str] of python module that uses DefaultLogger, e.g. [..]/module.py
        :param log_file: [str] of log file, e.g. [..]/logs.log
        :param level:    [str] 'debug', 'info' or 'warning (minimum logging level)
        :param mode:     [str] 'w' (write) or 'a' (append)
        :created attr: logger [python logging logger]
        """

        name = path.split("/")[-1]
        self.logger = logging.getLogger(name)
        self.logger.propagate = (
            level == "debug"
        )  # only shows console output if level is 'debug'

        # log file
        self.filename = log_file if log_file else self.default_log_file

        # level
        self.level = level.upper() if level else None
        if self.level:
            self.logger.setLevel(self.level)

        # add file handler
        if (
            len(self.logger.handlers) == 0
        ):  # avoid adding same file handler multiple times
            f_handler = logging.FileHandler(filename=self.filename, mode=mode)
            f_handler.setFormatter(
                logging.Formatter(self.custom_format, datefmt=self.custom_datefmt)
            )
            if self.level:
                f_handler.setLevel(self.level)
            self.logger.addHandler(f_handler)

        if mode == "w":
            self.log_debug(f"ACTIVATE FILE LOGGER {name} w/ PATH = {self.filename}")
        elif mode == "a":
            self.log_debug(f"APPEND TO FILE LOGGER {name} w/ PATH = {self.filename}")

    def clear(self):
        """
        clear file at log_file
        """
        with open(self.filename, "w") as f:
            f.write("")

    def log_debug(self, *args):
        """
        debug: log & print
        """
        msg = " ".join([str(elem) for elem in args])
        self.logger.debug(msg)  # log
        if self.level in ["DEBUG"]:
            print("DEBUG -----", msg)  # print

    def log_info(self, *args):
        """
        info: log & print
        """
        msg = " ".join([str(elem) for elem in args])
        self.logger.info(msg)  # log
        if self.level in ["DEBUG", "INFO"]:
            print("INFO ------", msg)  # print

    def log_warning(self, *args):
        """
        warning: log & print
        """
        msg = " ".join([str(elem) for elem in args])
        self.logger.warning(msg)  # log
        if self.level in ["DEBUG", "INFO", "WARNING"]:
            print("WARNING ---", msg)  # print
