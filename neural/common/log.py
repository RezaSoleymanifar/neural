import logging
import os

# Get the directory of the current script
log_dir = os.path.dirname(os.path.abspath(__file__))

# Create a subdirectory named "logs" to store the log files
log_dir = os.path.join(log_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the log file name
FILE_NAME = 'neural.log'

# Set up the basic logging configuration with the specified file name and logging level
logging.basicConfig(filename=os.path.join(
    log_dir, FILE_NAME), level=logging.DEBUG)

# Create a logger named 'neural'
logger = logging.getLogger('neural')

# Log a message indicating that the logger is configured
logger.info('neural logger configured.')

# Create a console handler for logging messages to the console and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for the console handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Set the formatter for the console handler
console_handler.setFormatter(formatter)

# Add the console handler to the 'neural' logger
logger.addHandler(console_handler)
