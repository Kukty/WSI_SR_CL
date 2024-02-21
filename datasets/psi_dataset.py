from abc import ABC
from typing import Iterable


class PSIDatasetParams(ABC):
    pass


class PSIDataset(ABC):
    def n_classes(self) -> int | None:
        return None

    def __iter__(self) -> Iterable:
        pass

    def __len__(self) -> int:
        pass

    def close(self) -> None:
        pass
