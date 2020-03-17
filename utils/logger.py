
import logging

from os.path import abspath, dirname, join
import sys
BASE_DIR = abspath(dirname(dirname(__file__)))
sys.path.append(BASE_DIR)

custom_datefmt = '%Y/%m/%d %H:%M:%S'
custom_format = '%(asctime)s - %(name)s - %(levelname)s -   %(message)s'
logging.basicConfig(
    level=logging.INFO,
    datefmt=custom_datefmt,
    format=custom_format,
    filename=f'{BASE_DIR}/logs.log',
    filemode='a',
)
logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)


def get_file_logger(_log_path):
    _absolute_log_path = join(BASE_DIR, _log_path)
    logger.info(f'ACTIVATE FILE LOGGER w/ PATH = {_absolute_log_path}')

    f_handler = logging.FileHandler(filename=_absolute_log_path, mode='w')
    # f_handler.setFormatter(logging.Formatter(custom_format, datefmt=custom_datefmt))
    logger.addHandler(f_handler)

    return logger
