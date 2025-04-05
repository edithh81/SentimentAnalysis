import sys
import traceback
import logging
import os
from datetime import datetime

# Configure logging
debug_logger = logging.getLogger('airflow.task')
debug_logger.setLevel(logging.DEBUG)

# Add file handler to log to a separate file for debugging
log_dir = '/opt/airflow/logs/debug'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(file_handler)

def log_exception(task_name="unknown_task"):
    """
    Log exception with full traceback and environment information.
    Call this in exception handlers.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    error_msg = ''.join(tb_lines)
    
    debug_logger.error(f"TASK ERROR in {task_name}:\n{error_msg}")
    
    # Log Python and library versions
    import torch
    import torchtext
    versions_info = (
        f"Python version: {sys.version}\n"
        f"PyTorch version: {torch.__version__}\n"
        f"TorchText version: {torchtext.__version__}\n"
    )
    debug_logger.error(f"Environment versions:\n{versions_info}")
    return error_msg

def trace_function(func):
    """
    Decorator to trace function calls with detailed error logging.
    """
    def wrapper(*args, **kwargs):
        try:
            debug_logger.info(f"STARTING {func.__name__}")
            result = func(*args, **kwargs)
            debug_logger.info(f"COMPLETED {func.__name__}")
            return result
        except Exception as e:
            error_msg = log_exception(func.__name__)
            print(f"Error in {func.__name__}: {e}")
            print(f"See detailed log at: {log_file}")
            # Re-raise the exception
            raise
    return wrapper