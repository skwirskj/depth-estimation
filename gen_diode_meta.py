import json
import os
import random

diode_data_path = '/media/lance/HDD/diode_data/'

meta_data = {}

for split in ['train', 'val', 'test']:
    meta_data[split] = {}
    for env in ['indoors']:
        meta_data[split][env] = {}
        for scene in os.listdir(os.path.join(diode_data_path, split, env)):
            meta_data[split][env][scene] = {}
            for scan in os.listdir(os.path.join(diode_data_path, split, env, scene)):
                _, ext = os.path.splitext(scan)
                if ext == '.txt':
                    continue

                meta_data[split][env][scene][scan] = []
                for f in os.listdir(os.path.join(diode_data_path, split, env, scene, scan)): 
                    name, ext = os.path.splitext(f)
                    if ext != '.png':
                        continue
                    if split == 'train':
                        if random.random() > 0.5:
                            continue

                    meta_data[split][env][scene][scan].append(name)

with open('new_meta.json', 'w') as outfile:
    json.dump(meta_data, outfile)
                    
                    