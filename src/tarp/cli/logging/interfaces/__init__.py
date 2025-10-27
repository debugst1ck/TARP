from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Abstract base interface for all loggers."""

    @abstractmethod
    def debug(self, message: str, *args, **kwargs): ...

    @abstractmethod
    def info(self, message: str, *args, **kwargs): ...

    @abstractmethod
    def warning(self, message: str, *args, **kwargs): ...

    @abstractmethod
    def error(self, message: str, *args, **kwargs): ...

    @abstractmethod
    def critical(self, message: str, *args, **kwargs): ...

    @abstractmethod
    def exception(self, message: str, *args, exc_info=True, **kwargs): ...
