import logging
import os
import datetime

LOG_LEVEL_DICT = {
    "logging.debug": logging.DEBUG,
    "logging.info": logging.INFO,
    "logging.warning": logging.WARNING,
    "logging.error": logging.ERROR,
    "logging.critical": logging.CRITICAL
}

DEFAULT_LOG_FORMAT = '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s'
DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT.get(os.environ.get('DEFAULT_LOG_LEVEL', 'logging.INFO').lower(), logging.INFO)


log_folder = './log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_folder, f"log_{timestamp}.log")

def setup_logging(logger_name, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging("my_logger")