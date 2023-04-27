import logging
from logging.handlers import RotatingFileHandler
from neural.common.constants import LOG_PATH, MAX_LOG_SIZE, LOG_BACKUP_COUNT


# Set up a rotating file handler with the specified file name, maximum size, 
# and backup count
if LOG_PATH is None:
    file_handler = logging.NullHandler()

else:
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=MAX_LOG_SIZE,
        backupCount=LOG_BACKUP_COUNT)

# Set the logging level for the file handler
file_handler.setLevel(logging.INFO)

# Create a formatter for the file handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the file handler
file_handler.setFormatter(formatter)

# Create a logger named 'neural'
logger = logging.getLogger('neural')

# Log a message indicating that the logger is configured
logger.info('neural logger configured.')

# Add the file handler to the 'neural' logger
logger.addHandler(file_handler)

# Create a console handler for logging messages to the console and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the formatter for the console handler
console_handler.setFormatter(formatter)

# Add the console handler to the 'neural' logger
logger.addHandler(console_handler)
