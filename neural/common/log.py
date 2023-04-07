import logging
import os

log_dir = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(log_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

FILE_NAME = 'alpacarl.log'
logging.basicConfig(filename=os.path.join(log_dir, FILE_NAME), level=logging.DEBUG)
logger = logging.getLogger('neural')
logger.info('AlpacaRL logging configured.')

# create console handler and set level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# add formatter to console handler
console_handler.setFormatter(formatter)

# add console handler to logger
logger.addHandler(console_handler)