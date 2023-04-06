import logging
import os

# Get the absolute path of the log file
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'test.log')

# Create a logger with the __name__ of the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for the logger
fh = logging.FileHandler(LOG_FILE_PATH)
fh.setLevel(logging.DEBUG)

# Create a stream handler for the logger to output log messages to the console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter for the log messages
formatter = logging.Formatter('%(levelname)s - %(message)s')

# Set the formatter for the file handler
fh.setFormatter(formatter)

# Set the formatter for the stream handler
ch.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(fh)

# Add the stream handler to the logger
logger.addHandler(ch)
