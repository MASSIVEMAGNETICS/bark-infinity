import functools
import sys
import time
from .config import logger


def retry_on_exception(max_retries=3, delay=1.0):
    """Decorator to retry a function if it raises an exception."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        "Attempt %s/%s failed in %s: %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        e,
                    )
                    time.sleep(delay)
            logger.error("%s failed after %s attempts", func.__name__, max_retries)
            return None

        return wrapper

    return decorator


def set_global_exception_logger():
    """Log uncaught exceptions using the project logger."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception
