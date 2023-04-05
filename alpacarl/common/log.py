import logging
import os

# get log directory path from installed package
log_dir = os.path.dirname(os.path.abspath(__file__))

# Create subdirectory for log files
log_dir = os.path.join(log_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, "alpacarl.log"), level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info('Logging configured.')

# create console handler and set level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# add formatter to console handler
console_handler.setFormatter(formatter)

# add console handler to logger
logger.addHandler(console_handler)