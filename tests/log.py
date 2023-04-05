import logging
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), 'test.log')

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create file handler and set level to debug
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# add formatter to file handler
fh.setFormatter(formatter)

# add file handler to logger
logger.addHandler(fh)
