import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(level='INFO', log_file=None, log_to_stdout=True, name=None):
    """
    Set up logging configuration
    
    Args:
        level: Logging level
        log_file: Log file path
        log_to_stdout: Whether to log to stdout
        name: Logger name
    
    Returns:
        logging.Logger: Logger object
    """
    # Map string log level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add stdout handler if requested
    if log_to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger