from torch.utils.data import Dataset

from datasets.psi_dataset import PSIDataset, PSIDatasetParams
from datasets.psi_parallel_dataset import PSIDatasetParallel


class TorchPSIDataset(Dataset):
    def __init__(
        self,
        ds_constructor: PSIDataset,
        ds_params: PSIDatasetParams,
        n_procs: int,
        queue_size: int,
        transform=None,
        filter=None,
    ) -> None:
        self.transform = transform
        self.filter = filter
        self._loader = PSIDatasetParallel(
            ds_constructor,
            ds_params,
            n_procs,
            queue_size,
        )
        self._n_classes = self._loader._n_classes
        self._iterable = iter(self._loader)

    def __len__(self):
        return self._loader.approx_len()

    def __getitem__(self, index):
        d = next(self._iterable)
        if self.filter:
            while not self.filter(d):
                d = next(self._iterable)
        if self.transform:
            d = self.transform(d)
        return d

    def close(self):
        self._loader.close()
