import logging
import os

# get log directory path from installed package
log_dir = os.path.dirname(os.path.abspath(__file__))

# Create subdirectory for log files
log_dir = os.path.join(log_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, "alpacarl.log"), level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Logging configured')