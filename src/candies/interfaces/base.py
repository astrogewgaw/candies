from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from autoregistry import Registry
from typing_extensions import Self

from candies.base import Candy


@dataclass
class Interface(Registry, recursive=False, suffix="Interface"):

    nf: int
    nt: int
    df: float
    dt: float
    fh: float
    nbits: int
    extras: dict

    @property
    def bw(self) -> float:
        return self.nf * self.df

    @property
    def fl(self) -> float:
        return self.fh - self.bw + (0.5 * self.df)

    @abstractmethod
    def slice(self, candy: Candy) -> tuple[float, float, np.ndarray]:
        pass


@dataclass
class FileInterface(Interface, suffix="File"):

    fn: str | Path

    @classmethod
    @abstractmethod
    def load(cls, fn: str | Path) -> Self:
        pass


@dataclass
class LiveInterface(Interface, suffix="Live"):

    @classmethod
    @abstractmethod
    def load(cls) -> Self:
        pass
