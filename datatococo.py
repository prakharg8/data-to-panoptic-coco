"""
STEPS:
    (See https://github.com/google-research/deeplab2/blob/main/g3doc/setup/getting_started.md for more instructions.)
    1. Install all dependencies and deeplab2 library. (Instructions and requirements on https://github.com/google-research/deeplab2/blob/main/g3doc/setup/installation.md)
        a. After installing everything use deeplab2/compile.sh gpu to compile and test everything.
        b. If you get gpu errors, create symlink between {pathtotensorflow}/tensorflow/include/third_party/gpus/cuda and {pathtocuda}.
        c. If you still get an error create symlink between /usr/local/cuda/lib64/libcudart.so and /usr/lib/libcudart.so
    2. Run this script to convert dataset to coco panoptic format.
    3. Edit deeplab2/data/build_coco_data.py according to the dataset and use it to convert the dataset to tfrecords.
        a. Remove 2017 from everywhere in deeplab2/data/build_coco_data.py and then use it.

        python deeplab2/data/build_coco_data.py --coco_root={pathtodataset} --output_dir={pathtofinaldataset}

    4. Copy the provided config file to the root directory of the project.
    5. Change train-dataset file pattern in config file to {final-dataset-directory}/train*.tfrecord.
    6. Change val-dataset file pattern in config file to {final-dataset-directory}/val*.tfrecord.
    7. Change deeplab2/data/dataset.py to include the dataset. (Edit the Coco dataset settings. Change the number of classes and split sizes)
    8. Create checkpoints directory and download desired intitial checkpoint from https://github.com/google-research/deeplab2/blob/main/g3doc/projects/imagenet_pretrained_checkpoints.md.
    9. Use deeplab2/trainer/train.py to start training in desired mode and use python deeplab2/trainer/train.py --help to see options.
        
        python project/deeplab2/trainer/train.py --config_file=config.textproto --mode=train_and_eval --model_dir=./checkpoints --num_gpus=1

"""


# all annotation jsons
# change accordingly
ann_files = [
    './data/prakhar/1/1W_LBSS-C1_HSM_NVR-2_nothuman-export.json', 
    './data/prakhar/2/1W_LBSS-C2_2-export.json', 
    './data/prakhar/3/1W_LBSS_C2_3-export.json', 
    './data/prakhar/31/fmlbss-export.json', 
    './data/prakhar/32/lbssc2-export.json', 
    './data/prakhar/33/lbssc3-export.json', 
    './data/prakhar/34/RM#6-C1-export.json', 
    './data/prakhar/35/RM#6-C2-export.json', 
    './data/prakhar/36/rmlbssc2-export.json', 
    './data/prakhar/4/24/1W_LBSS_C2_4-export.json', 
]
# paths to all images 
# change accordingly
imgs = [
    './data/prakhar/1/', 
    './data/prakhar/2/', 
    './data/prakhar/3/', 
    './data/prakhar/31/', 
    './data/prakhar/32/', 
    './data/prakhar/33/', 
    './data/prakhar/34/', 
    './data/prakhar/35/', 
    './data/prakhar/36/', 
    './data/prakhar/4/24/', 
]

# utils for converting data to coco panoptic format
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py

import pycocotools._mask as _mask

def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]

def area(rleObjs):
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]

class IdGenerator():
    '''
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    '''
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

# data to coco panoptic format
# https://cocodataset.org/#format-data:~:text=4.%20Panoptic%20Segmentation
# https://github.com/google-research/deeplab2/blob/main/g3doc/setup/coco.md

import cv2
import os
import os.path as osp
import json
from tqdm import tqdm
import numpy as np

def convert_data_to_coco(ann_files, img_dirs, out_file_train, out_file_val, out_img_train, out_img_val, split=0.8):
    if not osp.isfile(out_img_train): os.makedirs(out_img_train, exist_ok=True)
    if not osp.isfile(out_img_val): os.makedirs(out_img_val, exist_ok=True)

    dic = {
        'panel': 1,
        'red_panel': 2
    }
    categories = [
        {
            'id':1, 
            'name':'panel', 
            'isthing':1, 
            'color':[255,0,0]
            }, 
        {
            'id':2, 
            'name':'red_panel', 
            'isthing':1, 
            'color':[0,255,0]
            },
    ]
    categories_gen = {el['id']: el for el in categories}
    OFFSET = 256

    # test
    images = []
    annotations = []
    for (ann_file, imgs) in zip(ann_files, img_dirs):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        
        total = len(data_infos['assets'].values())

        for idx, v in tqdm(zip(range(int(total * split)), data_infos['assets'].values())):
            obj_count = 1
            filename = v['asset']['name']
            img_path = osp.join(imgs, filename)

            if not osp.isfile(img_path): continue

            id_generator = IdGenerator(categories_gen)

            height = 1080
            width = 1920

            pan_format = np.zeros(shape=(height, width, 3), dtype=np.int32)

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))
            
            segments_info  = []
            # iterate over all instances in an image
            for el, obj in enumerate(v['regions']):
                if obj['tags'][0] == 'dicey': continue
                
                obj_cat = dic[obj['tags'][0]]
                segment_id, color = id_generator.get_id_and_color(obj_cat)
                pan_id = obj_cat * OFFSET + obj_count

                all_points = []
                for point in obj["points"]:
                    all_points.append([point["x"], point["y"]])
                cv2.fillPoly(pan_format, [np.array(all_points, dtype=np.int32)], color=[pan_id // OFFSET // OFFSET, pan_id // OFFSET, pan_id % OFFSET])
                mask = np.zeros(shape=[height, width, 1], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(all_points, dtype=np.int32)], color=(1))

                px = [a['x'] for a in obj['points']]
                py = [a['y'] for a in obj['points']]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))
                obj_count += 1

                segment_info = dict(
                    id=pan_id,
                    category_id=obj_cat,
                    area=int(area(encode(np.asfortranarray(mask)))),
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    iscrowd=0,
                )
                segments_info.append(segment_info)

            cv2.imwrite(osp.join(out_img_train, filename.replace('jpg', 'png')), pan_format)
            annotation = dict(
                image_id=idx,
                file_name=filename.replace('jpg', 'png'),
                segments_info=segments_info,
            )
            annotations.append(annotation)

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    with open(out_file_train, 'w') as outfile:
        json.dump(coco_format_json, outfile)

    # val
    images = []
    annotations = []
    for (ann_file, imgs) in zip(ann_files, img_dirs):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        
        total = len(data_infos['assets'].values())
        
        for idx, v in tqdm(zip(range(int(total * split), total), data_infos['assets'].values())):
            obj_count = 1
            filename = v['asset']['name']
            img_path = osp.join(imgs, filename)

            if not osp.isfile(img_path): continue

            id_generator = IdGenerator(categories_gen)

            height = 1080
            width = 1920

            pan_format = np.zeros(shape=(height, width, 3), dtype=np.int32)
            
            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))

            segments_info = []
            for obj in v['regions']:
                if obj['tags'][0] == 'dicey': continue

                obj_cat = dic[obj['tags'][0]]
                segment_id, color = id_generator.get_id_and_color(obj_cat)
                pan_id = obj_cat * OFFSET + obj_count

                all_points = []
                for point in obj["points"]:
                    all_points.append([point["x"], point["y"]])
                cv2.fillPoly(pan_format, [np.array(all_points, dtype=np.int32)], color=[pan_id // OFFSET // OFFSET, pan_id // OFFSET, pan_id % OFFSET])
                mask = np.zeros(shape=[height, width, 1], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(all_points, dtype=np.int32)], color=(1))
                
                px = [a['x'] for a in obj['points']]
                py = [a['y'] for a in obj['points']]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))
                obj_count += 1

                segment_info = dict(
                    id=pan_id,
                    category_id=obj_cat,
                    area=int(area(encode(np.asfortranarray(mask)))),
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    iscrowd=0,
                )
                segments_info.append(segment_info)

            cv2.imwrite(osp.join(out_img_val, filename.replace('jpg', 'png')), pan_format)
            annotation = dict(
                image_id=idx,
                file_name=filename.replace('jpg', 'png'),
                segments_info=segments_info,
            )
            annotations.append(annotation)

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    with open(out_file_val, 'w') as outfile:
        json.dump(coco_format_json, outfile)

split = 0.8
convert_data_to_coco(
    ann_files, # all annotation files
    imgs, # all image folders
    './dataset/annotations/panoptic_train.json', # destination file for train annotations 
    './dataset/annotations/panoptic_val.json', # destination file for val annotations
    './dataset/annotations/panoptic_train', # destination folder of generated masks for training
    './dataset/annotations/panoptic_val', # destination folder of generated masks for validation
    split=0.8 # train-validation split
)

import shutil

dest_train = './dataset/train'
dest_val = './dataset/val'
if not osp.isdir(dest_train): os.makedirs(dest_train)
if not osp.isdir(dest_val): os.makedirs(dest_val)

for ann_file, img in zip(ann_files, imgs):
    with open(ann_file, 'r') as f:
        data_infos = json.load(f)
        
    total = len(data_infos['assets'].values())

    for idx, v in tqdm(zip(range(int(total * split)), data_infos['assets'].values())):
        filename = v['asset']['name']
        if not osp.isfile(osp.join(img, filename)): continue
        shutil.copy(osp.join(img, filename), osp.join(dest_train, filename))

    for idx, v in tqdm(zip(range(int(total * split), total), data_infos['assets'].values())):
        filename = v['asset']['name']
        if not osp.isfile(osp.join(img, filename)): continue
        shutil.copy(osp.join(img, filename), osp.join(dest_val, filename))

"""
Installation steps used on colab.

%cd /content
!rm -rf project
!mkdir project
%cd project
!git clone https://github.com/google-research/deeplab2
!pip install tensorflow==2.6 keras==2.6
!sudo apt-get install protobuf-compiler
!pip install pillow cython
!git clone https://github.com/tensorflow/models.git
!git clone https://github.com/cocodataset/cocoapi.git
%cd cocoapi/PythonAPI
!make
%cd ../../
%env PYTHONPATH=$PYTHONPATH:/content/project:/content/project/models:/content/project/cocoapi/PythonAPI
%cd /usr/local/lib/python3.7/dist-packages/tensorflow/
!mkdir ./include/third_party/gpus
!ls ./include/third_party/
%cd ./include/third_party/gpus
!ln -s /usr/local/cuda ./cuda
%cd /content/project
!sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so
!bash deeplab2/compile.sh gpu
!bash deeplab2/compile.sh
!ls deeplab2
%cd /content
"""
