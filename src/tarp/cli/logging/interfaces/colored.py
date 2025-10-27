import logging
import os
import sys
from tarp.cli.logging.interfaces import BaseLogger
import io


class ColoredLogger(BaseLogger):
    """Logger that prints color-coded messages to console and file."""

    _COLORS = {
        "DEBUG": "\033[96m",
        "INFO": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    _RESET = "\033[0m"

    class _ColoredFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = ColoredLogger._COLORS.get(record.levelname, ColoredLogger._RESET)
            message = super().format(record)
            return f"{color}{message}{ColoredLogger._RESET}"

    def __init__(self, log_dir: str = "logs", name: str = "colored_logger"):
        os.makedirs(log_dir, exist_ok=True)
        self._logger = logging.getLogger(name)

        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            logging.addLevelName(logging.WARNING, "WARN")

            if hasattr(sys.stdout, "buffer"):  # True in normal terminal
                safe_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            else:
                # For environments like IPython/Jupyter that lack .buffer
                safe_stream = sys.stdout  # already UTF-8 capable in most modern setups
            file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")

            console_handler = logging.StreamHandler(safe_stream)
            console_handler.setFormatter(
                self._ColoredFormatter("[%(levelname)s]\t%(asctime)s - %(message)s")
            )
            file_handler.setFormatter(
                logging.Formatter("[%(levelname)s]\t%(asctime)s - %(message)s")
            )

            self._logger.addHandler(console_handler)
            self._logger.addHandler(file_handler)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, exc_info=True, **kwargs):
        self._logger.exception(message, *args, exc_info=exc_info, **kwargs)
