"""
An example of using psimage package to extract patches from
whole slide histological image (PATH-DT-MSU WSS1 dataset)
in accordance with polygonal annotation stored in json file.
Patches can be extracted either in a dence way with fixed stride
or in a random way.
"""
from tqdm import tqdm 
import json
from pathlib import Path
from psimage.patches import Patch
import numpy as np
import traceback
from psimage.base.patch_generators import PatchGenerator
from psimage import PSImage
from shapely.geometry import Polygon
from pathlib import Path
from psimage import PSImage
from psimage.base.anno_visualizer import AnnoDescription


ds_path = Path("/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/test")
anno_path = ds_path

out_path_4x = Path("data_v2.5/val_40x")
out_path_1x = Path("data_v2.5/val_10x")
out_path_2x = Path("data_v2.5/val_20x")

out_path_4x.mkdir(exist_ok=True, parents=True)
out_path_2x.mkdir(exist_ok=True, parents=True)
out_path_1x.mkdir(exist_ok=True, parents=True)


psi_files = sorted(ds_path.glob('*.psi'))
json_files = sorted(anno_path.glob('*.json'))
file_pairs = list(zip(psi_files, json_files))

for psim,anno_json in file_pairs:
    print(psim,anno_json)


    
# ```python
import concurrent.futures

def process_file(psim,anno_json):
    k = 0 
    strr = str(psim).split('/')

    dataset = strr[-3] +'_'+ strr[-1][:-4]
    psim = PSImage(psim)
    anno_json = open(anno_json)
    for anno in tqdm((json.load(anno_json))):
        anno_cls = anno["class"]
        out_path_class_4x =  out_path_4x / anno_cls
        out_path_class_2x =  out_path_2x / anno_cls
        out_path_class_1x =  out_path_1x / anno_cls
        out_path_class_2x.mkdir(exist_ok=True, parents=True)
        out_path_class_1x.mkdir(exist_ok=True, parents=True)
        out_path_class_4x.mkdir(exist_ok=True, parents=True)
        
        # print(anno_cls)
        anno_polygon = np.array(anno["vertices"], dtype=np.float64)
        # for base_size*base_size image , get from layer 4 
        # for base_size*2 image , get from layer 2
        # for base_size*4 image, get frim layer 1 
        base_size = 112 

        if strr[-2] == 'train':
            
            return NotImplementedError
                            
        else:
            
            try:
                for i, patch in enumerate(
                    psim.patch_gen_dense(base_size, stride=base_size, region=anno_polygon,layer=4)
                    ):
                    patch_name = f"{dataset}_{k+1}_{anno_cls}_{i+1}.jpg"
                    patch.to_image().save(out_path_class_1x / patch_name)
                                    # for 2x data 
                    (y,x) = patch.pos_y*2,patch.pos_x*2
                    patch_size = base_size*2
                    region = psim.get_region_from_layer(2,
                                        (y, x),
                                        (y + patch_size, x + patch_size),)
                    patch_2x = Patch(
                        layer=2,
                        pos_x=x,
                        pos_y=y,
                        patch_size=patch_size,
                        data=region,
                    )
                    patch_2x.to_image().save(out_path_class_2x / patch_name)
                    # for 4x data
                    (y,x) = patch.pos_y*4,patch.pos_x*4
                    patch_size = base_size*4 
                    region = psim.get_region_from_layer(1,
                                        (y, x),
                                        (y + patch_size, x + patch_size),)
                    patch_4x = Patch(
                        layer=1,
                        pos_x=x,
                        pos_y=y,
                        patch_size=patch_size,
                        data=region,
                    )
                    patch_4x.to_image().save(out_path_class_4x / patch_name)
                    k=k+1
            except Exception as e:
                print(traceback.format_exc())
            
    anno_json.close()
    psim.close()

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for psim,anno_json in file_pairs:
                executor.submit(process_file,psim,anno_json)
