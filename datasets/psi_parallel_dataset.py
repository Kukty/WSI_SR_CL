import time
from multiprocessing import Process, Queue, SimpleQueue
from typing import Iterable

import numpy as np

from datasets.psi_dataset import PSIDataset, PSIDatasetParams


class PSIDatasetParallel:
    def __init__(
        self,
        ds_constructor: PSIDataset,
        ds_params: PSIDatasetParams,
        n_procs: int,
        queue_size: int,
    ) -> None:
        self.data_queue = Queue(maxsize=queue_size)
        self.control_queue = SimpleQueue()

        print("Creating PSIDatasetParallel instance...")

        self.processes = []
        for i in range(n_procs):
            quiet = i != 0
            proc = Process(
                target=PSIDatasetParallel._single_loader,
                args=(
                    ds_constructor,
                    ds_params,
                    quiet,
                    self.data_queue,
                    self.control_queue,
                ),
            )
            self.processes.append(proc)
            proc.start()
        print("\t all processes started")
        self._ds_len = None
        self._n_classes = None
        for _ in range(n_procs):
            msg = self.control_queue.get()
            self._ds_len = msg["ds_len"]
            self._n_classes = msg["n_classes"]

    def _single_loader(
        ds_constructor,
        ds_params: PSIDatasetParams,
        quiet: bool,
        data_queue: Queue,
        control_queue: SimpleQueue,
    ):
        ds = ds_constructor(ds_params, quiet=quiet)
        control_queue.put({"ds_len": len(ds), "n_classes": ds.n_classes()})
        time.sleep(0.1)
        g = iter(ds)
        while True:
            if not control_queue.empty():
                msg = control_queue.get()
                if msg == "stop":
                    ds.close()
                    break
                else:
                    control_queue.put(msg)
            data_queue.put(next(g))

    def approx_len(self):
        return self._ds_len

    def n_classes(self):
        return self._n_classes

    def __iter__(self) -> Iterable[tuple[np.ndarray, int]]:
        while True:
            try:
                yield self.data_queue.get_nowait()
            except Exception:
                continue

    def close(self) -> None:
        print("Closing data loader...")
        for _ in range(len(self.processes)):
            self.control_queue.put("stop")
        print("\t stop signals sent")

        self.control_queue.close()
        print("\t control queue closed")
        self.data_queue.close()
        print("\t data queue closed")

        for proc in self.processes:
            try:
                proc.join(0.1)
                proc.close()
            except Exception:
                proc.kill()

        time.sleep(0.2)
        for proc in self.processes:
            proc.close()

        print("\t all processes closed")
        print("PSIDatasetParallel instance closed successfully.")
