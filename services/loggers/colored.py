import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[96m",
        "INFO": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


class ColoredLogger:
    _logger = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """
        Ensure only one logger instance exists (singleton).
        Logging library does follow singleton pattern
        but this is a wrapper for extending features later on.
        """
        if cls._logger is None:
            logger = logging.getLogger("ColoredLogger")
            logger.setLevel(logging.DEBUG)
            
            logging.addLevelName(logging.WARNING, "WARN")

            if not logger.handlers:  # only add handler once
                handler = logging.StreamHandler()
                formatter = ColoredFormatter(
                    "[%(levelname)s]\t%(asctime)s - %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            cls._logger = logger
        return cls._logger

    # Convenience static-like methods
    @classmethod
    def debug(cls, msg, *args, **kwargs):
        cls._get_logger().debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg, *args, **kwargs):
        cls._get_logger().info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg, *args, **kwargs):
        cls._get_logger().warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        cls._get_logger().error(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg, *args, **kwargs):
        cls._get_logger().critical(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg, *args, exc_info=True, **kwargs):
        cls._get_logger().exception(msg, *args, exc_info=exc_info, **kwargs)
