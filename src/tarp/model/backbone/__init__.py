from torch import nn
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional


class Encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def encode(
        self, sequence: Tensor, attention_mask: Optional[Tensor] = None, return_sequence: bool = False
    ) -> Tensor:
        pass

    @property
    @abstractmethod
    def encoding_size(self) -> int:
        pass


class FrozenModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def freeze(self):
        pass

    @abstractmethod
    def unfreeze(self):
        pass
