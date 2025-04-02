from types_ import *
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, input: Tensor) -> Tensor:
        raise NotImplementedError
