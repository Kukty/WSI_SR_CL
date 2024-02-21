import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from psimage.base.exceptions import PSImageException
from psimage.base.patch_generators import PatchGenerator
from psimage.image import PSImage
from shapely import Polygon
from psimage.patches import Patch
from datasets.psi_dataset import PSIDatasetParams


@dataclass
class PSIRegionSrClsDatasetParams(PSIDatasetParams):
    """Class that controls parameters of PSIRegionSRClsDataset.

    Args:
        path: path to directory with images and annotations.

        patch_size: size of extracted patches.

        layer: layer to extract patches from.

        region_intersection: minimum intersection of extracting
        regionpatch with annotated region.

        n_patches_from_one_region: number of patches continually
        extracted from one region.

        balance_coeff: coefficient of class balance, controlling the
        choice of region to extract patch from for the chosen class.
        If it is 0, than all regions are chosen with the same probability.
        It if is 1, that the region is chosen proportionally to it's area
        (bigger area - bigger probability).
        It if is -1, that the region is chosen inversly proportionally to
        it's area (bigger area - smaller probability).
        If it is in range (-1, 0) or (0, 1) then the probalities are avaraged
        as a linear combination between border cases. Closer to 0 - closer to
        uniform distribution. Closer to -1 - closer to the distribution of
        case -1, closer to 1 - closer to the distribution of case 1.

        hr_times: hr_img is *hr_times* bigger than lr_img, e.g lr_img is patch_size x patch_size, 
        then the hr_img is patch_size * hr_times x  patch_size * hr_times
    """

    path: Path
    patch_size: int
    layer: int
    region_intersection: float
    n_patches_from_one_region: int = 10
    balance_coeff: float = 0.5,
    hr_times: int = 2,
    annotations_path: Path = None 


@dataclass
class RegionAnnotation:
    file_name: str
    region_idx: int
    class_: str
    vertices: np.ndarray
    area: float = None
    random_generator: PatchGenerator = None

    def __post_init__(self):
        if self.area is None:
            self.area = Polygon(self.vertices).area

    def __str__(self) -> str:
        return (
            f"Region [{self.file_name}, {self.region_idx}, "
            f"{self.class_}, {self.vertices.shape}, {round(self.area, 0)}]"
        )


class PSIRegionSrClsDataset:

    """PSIDataset for the task of classification

    This dataset extracts random patches from the annotated regions of
    PSImages. All annotated regions in the dataset are maped according
    to the classes. Each time a random class is chosen and a random
    corresponding region. After this for the reason of performance a number
    of random patches are extracted from this region.
    """

    def __init__(self, params: PSIRegionSrClsDatasetParams, quiet=False) -> None:
        self.params = params
        self.quiet = quiet
        self.patch_size = params.patch_size
        if not self.quiet:
            print("Reading annotation files...")
        if params.annotations_path is None :
            anno_files = sorted(
                [
                    f
                    for f in params.path.iterdir() 
                    if f.is_file()  
                    and f.suffix == ".json"
                    and "_v2.5.json" in str(f)
                ]
            )
        else:
            anno_files = sorted(
                [
                    f
                    for f in params.annotations_path.iterdir() 
                    if f.is_file()  
                    and f.suffix == ".json"
                    and "_v2.5.json" in str(f)
                ]
            )
        self._psims = []
        self._regions: dict[str, RegionAnnotation] = dict()
        for anno_file in anno_files:
            # TODO in the future change file name to the canonical version!
            file_name = anno_file.name.split("_v2.5")[0]
            psim = PSImage(params.path / f"{file_name}.psi")
            self._psims.append(psim)
            with open(anno_file) as f:
                for i, a in enumerate(json.load(f)):
                    cls = a["class"]
                    vertices = np.array(a["vertices"], dtype=np.float64)
                    try:
                        random_generator = psim.patch_gen_random(
                            patch_size=params.patch_size,
                            region=vertices,
                            region_intersection=params.region_intersection,
                            layer=params.layer,
                        )
                        region = RegionAnnotation(
                            file_name=file_name,
                            region_idx=i,
                            class_=cls,
                            vertices=vertices,
                            random_generator=random_generator,
                        )
                        if cls not in self._regions:
                            self._regions[cls] = [region]
                        else:
                            self._regions[cls].append(region)
                    except PSImageException:
                        continue

        self._calc_areas()
        self._calc_weights(self.params.balance_coeff)
        self._class_labels = sorted(list(self._regions.keys()))
        self._label_to_idx = {v: k for k, v in enumerate(self._class_labels)}

    def n_classes(self):
        return len(self._class_labels)

    def _calc_areas(self):
        self._areas = {
            k: sum([i.area for i in v]) for k, v in self._regions.items()
        }

        areas_total = sum(self._areas.values())
        self._areas_n = {k: v / areas_total for k, v in self._areas.items()}

        if not self.quiet:
            print(f"Annotations for {len(self._regions)} classes found:")
            for k in self._areas:
                area_megapixels = round(self._areas[k] / 1e6, 2)
                area_prc = round(self._areas_n[k] * 100, 2)
                print(f"\t {k}: area {area_megapixels}M pixels ({area_prc}%)")

    def _calc_weights(self, balance_coeff: float):
        assert -1 <= balance_coeff <= 1
        self._weights = dict()
        for cls, regions in self._regions.items():
            areas = [r.area for r in regions]
            areas_inv = [1 / a for a in areas]
            w_proportional = np.array(areas) / sum(areas)
            w_inv_proportional = np.array(areas_inv) / sum(areas_inv)
            w_default = np.ones(len(regions), dtype=np.float64) / len(regions)

            if balance_coeff == 0:
                w = w_default
            elif balance_coeff > 0:
                delta = (w_proportional - w_default) * balance_coeff
                w = w_default + delta
                w = w / sum(w)
            elif balance_coeff < 0:
                delta = (w_inv_proportional - w_default) * (-balance_coeff)
                w = w_default + delta
                w = w / sum(w)

            self._weights[cls] = w

    def __iter__(self) -> Iterable[tuple[np.ndarray, int]]:
        while True:
            cls = np.random.choice(self._class_labels,p=np.array([0.2,0.1,0.2,0.2,0.3]))
            # cls = np.random.choice(self._class_labels)
            cls_idx = self._label_to_idx[cls]
            region = np.random.choice(
                self._regions[cls],
                p=self._weights[cls],
            )
            region_gen = region.random_generator
            for _ in range(self.params.n_patches_from_one_region):
                try:
                    # data = next(region_gen).data.astype(np.float32) / 255
                    data = next(region_gen)
                    lr_img = data.data.astype(np.float32) / 255
                    (y,x) = data.pos_y*self.params.hr_times,data.pos_x*self.params.hr_times
                    patch_size = self.patch_size*self.params.hr_times
                    region_index = int(region.file_name[-1]) -1 
                    psim = self._psims[region_index]
                    hr_img = psim.get_region_from_layer(self.params.layer // self.params.hr_times,
                                        (y, x),
                                        (y + patch_size, x + patch_size),)                    
                    hr_img = hr_img.astype(np.float32) / 255
                    lbl = np.zeros(len(self._class_labels), dtype=np.float32)
                    lbl[cls_idx] = 1
                    yield lr_img,hr_img, lbl
                except Exception:
                    continue

    def __len__(self):
        ps = self.patch_size * self.params.layer
        return int(sum(self._areas.values()) / (ps * ps))

    def close(self):
        for psim in self._psims:
            psim.close()
