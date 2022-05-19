import os 
import json
from shutil import copyfile

def mkdir(path, dst_dir):
    full = os.path.join(dst_dir, path)
    print(full)
    if not os.path.exists(full):
        print("Making path:", full)
        os.mkdir(full)

src_dir = "/media/lance/HDD/diode_data"
dst_dir = "/media/lance/HDD/diode_subset"

with open('/home/lance/eecs442/442-depth-estimation/new_meta.json') as f:
    data = json.load(f)

data = data['train']['indoors']

for scene in data:
    path = "train/indoors/"
    scene_path = os.path.join(path, scene)
    mkdir(scene_path, dst_dir)
    for scan in data[scene]:
        scan_path = os.path.join(scene_path, scan)
        mkdir(scan_path, dst_dir)

        for f in data[scene][scan]:
            file_path = os.path.join(scan_path, f)
            src_path = os.path.join(src_dir, file_path)
            dst_path = os.path.join(dst_dir, file_path)

            png_src = src_path + ".png"
            de_src = src_path + "_depth.npy"
            de_mask_src = src_path + "_depth_mask.npy"

            png_dst = dst_path + ".png"
            de_dst = dst_path + "_depth.npy"
            de_mask_dst = dst_path + "_depth_mask.npy"
            
            copyfile(png_src, png_dst)
            copyfile(de_src, de_dst)
            copyfile(de_mask_src, de_mask_dst)


