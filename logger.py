import os
import logging

# Create log directory if it doesn't exist
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, 'app.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')

# Create a logger
logger = logging.getLogger('thyroid_app')

def get_logger():
    return logger
