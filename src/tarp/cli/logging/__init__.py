from tarp.cli.logging.interfaces import BaseLogger
from tarp.cli.logging.interfaces.colored import ColoredLogger


class Console:
    """Unified facade for logging operations.

    Default backend is `ColoredLogger`.
    You can swap the backend using `Console.use()`.
    """

    _backend: BaseLogger = ColoredLogger()

    @classmethod
    def use(cls, backend: BaseLogger):
        """Replace the current backend at runtime."""
        cls._backend = backend

    # Delegate methods
    @classmethod
    def debug(cls, message, *args, **kwargs):
        cls._backend.debug(message, *args, **kwargs)

    @classmethod
    def info(cls, message, *args, **kwargs):
        cls._backend.info(message, *args, **kwargs)

    @classmethod
    def warning(cls, message, *args, **kwargs):
        cls._backend.warning(message, *args, **kwargs)

    @classmethod
    def error(cls, message, *args, **kwargs):
        cls._backend.error(message, *args, **kwargs)

    @classmethod
    def critical(cls, message, *args, **kwargs):
        cls._backend.critical(message, *args, **kwargs)

    @classmethod
    def exception(cls, message, *args, exc_info=True, **kwargs):
        cls._backend.exception(message, *args, exc_info=exc_info, **kwargs)
