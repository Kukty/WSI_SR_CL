from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from psimage.image import PSImage

from datasets.psi_dataset import PSIDataset, PSIDatasetParams


@dataclass
class PSIRandomSRDatasetParams(PSIDatasetParams):
    path: Path
    patch_size: int
    layer_src: int
    layer_trg: int
    block_size_in_patches: int = 15
    block_fill_prc: float = 0.3
    blocks_per_img: int = 10

    def __post_init__(self):
        assert self.layer_trg < self.layer_src


class PSIRandomSRDataset(PSIDataset):
    """PSIDataset for the task of super resolution

    This dataset extracts random patches from the random PSImage at source
    layer and correspondinig target layer. For the reason of performance it
    chooses random block (rather big fragment of an image) and aftewards
    extracts some random patches inside. The block extraction is performed
    several times for each image.
    """

    def __init__(self, params: PSIRandomSRDatasetParams, quiet=False) -> None:
        self.params = params
        self.quiet = quiet
        img_files = [
            i
            for i in params.path.iterdir()
            if i.is_file() and i.suffix == ".psi"
        ]
        self._psims = [PSImage(img_file) for img_file in img_files]
        sizes = [
            psim.layer_size(self.params.layer_src) for psim in self._psims
        ]
        area = sum([s[0] * s[1] for s in sizes])
        self._approx_len = int(area / (params.patch_size * params.patch_size))

    def __iter__(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        ps = self.params.patch_size
        # rescale coefficient
        rc = int(self.params.layer_src / self.params.layer_trg)
        while True:
            psim: PSImage = np.random.choice(self._psims)
            h, w = psim.layer_size(self.params.layer_src)
            blk_h = min(ps * self.params.block_size_in_patches, h)
            blk_w = min(ps * self.params.block_size_in_patches, w)
            blk_patches = int(
                blk_h * blk_w / (ps * ps) * self.params.block_fill_prc
            )
            for _ in range(self.params.blocks_per_img):
                # first extract a block from psimage
                blk_x = np.random.randint(0, w - blk_w)
                blk_y = np.random.randint(0, h - blk_h)
                blk = psim.get_region_from_layer(
                    self.params.layer_src,
                    (blk_y, blk_x),
                    (blk_y + blk_h, blk_x + blk_w),
                )
                # it is also more efficient to extract a corresponding
                # block at target layer
                blk_trg = psim.get_region_from_layer(
                    self.params.layer_trg,
                    (blk_y * rc, blk_x * rc),
                    (
                        blk_y * rc + blk_h * rc,
                        blk_x * rc + blk_w * rc,
                    ),
                )
                for _ in range(blk_patches):
                    # then extract patches from the block
                    x = np.random.randint(0, blk_w - ps)
                    y = np.random.randint(0, blk_h - ps)
                    patch_src = blk[y : y + ps, x : x + ps]

                    patch_trg = blk_trg[
                        y * rc : y * rc + ps * rc,
                        x * rc : x * rc + ps * rc,
                    ]
                    yield (
                        patch_src.astype(np.float32) / 255,
                        patch_trg.astype(np.float32) / 255,
                    )

    def __len__(self):
        return self._approx_len

    def close(self) -> None:
        for psim in self._psims:
            psim.close()
