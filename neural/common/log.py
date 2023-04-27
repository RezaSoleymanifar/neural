import logging
from logging.handlers import RotatingFileHandler
from neural.common.constants import LOG_PATH, MAX_LOG_SIZE, LOG_BACKUP_COUNT, LOG_LEVEL



# =========================setup logger=============================

# Create a logger named 'neural'
logger = logging.getLogger('neural')
# Set the logging level for the logger
logger.setLevel(LOG_LEVEL)

# =========================file/console handler=============================

# Create a rotating file handler with the specified file name, maximum size, 
# and backup count. If LOG_PATH is None, then set the file handler to a
# NullHandler, which will not log any messages to a file.

if LOG_PATH is not None:
    file_handler = RotatingFileHandler(
        filename=LOG_PATH,
        maxBytes=MAX_LOG_SIZE,
        backupCount=LOG_BACKUP_COUNT)
else:
    file_handler = logging.NullHandler()

# Set the logging level for the file handler
file_handler.setLevel(LOG_LEVEL)


# Create a console handler for logging messages to the console.
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)


# Add the file handler to the 'neural' logger
logger.addHandler(file_handler)
# Add the console handler to the 'neural' logger
logger.addHandler(console_handler)


# =========================logger formatter=============================

# Create a formatter for the file handler
formatter = logging.Formatter('%(levelname)s - %(message)s')

# Set the formatter for the file handler
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


