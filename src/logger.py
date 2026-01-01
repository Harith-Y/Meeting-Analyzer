"""
Logging configuration for the application
"""
import logging
import logging.config
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import LOGGING_CONFIG, LOGS_DIR


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Set up and return a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__ of the calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Get logger
    logger = logging.getLogger(name)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters and results.
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


# Create a default logger for the application
app_logger = setup_logger("lecture_transcription")
