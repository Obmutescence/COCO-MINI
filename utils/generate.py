import os
import json
import numpy as np
from shutil import copyfile
from pycocotools.coco import COCO

np.random.seed(42)

COCO_DATA_ROOT = "data/data103218"

COCO2017_METAINFO = {
    'classes':
    ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    # palette is a list of color tuples, which is used for visualization.
    'palette':
    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
        (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
        (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
        (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
        (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
        (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
        (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
        (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
        (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
        (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
        (246, 0, 122), (191, 162, 208)]
}

for train_val_str in ["train2017", "val2017"]:

    # --------------------------- for instances ---------------------------
    with open(f"{COCO_DATA_ROOT}/annotations/instances_{train_val_str}.json") as f:
        instances_trainval2017_dict = json.load(f)
    coco = COCO(f"{COCO_DATA_ROOT}/annotations/instances_{train_val_str}.json")

    img_id_list = []  # 存用来图ID
    for cls_ in COCO2017_METAINFO['classes']:

        # 获取当前类ID
        cls_id = coco.getCatIds(cls_)[0]
        # 通过当前类ID, 找到包含该类的所有图片ID
        imgIds = coco.getImgIds(catIds=[cls_id])
        # 打印含有当前类别的图片数量
        img_nums = len(imgIds)
        print(f"[CLS]: {cls_},\t[ID]: {cls_id},\t[NUMS]: {img_nums}")

        # 每类选择一张图
        img_id = np.random.choice(imgIds)
        img_id_list.append(img_id)

    ann_id_list = []  # 存标注ID
    for img_id in img_id_list:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_id_list += ann_ids

    images = []
    for img_info in instances_trainval2017_dict['images']:
        if img_info['id'] in img_id_list:
            images.append(img_info)

    annotations = []
    for ann_info in instances_trainval2017_dict['annotations']:
        if ann_info['id'] in ann_id_list:
            annotations.append(ann_info)

    instances_trainval2017_dict['images'] = images
    instances_trainval2017_dict['annotations'] = annotations

    with open(f"instances_{train_val_str}.json", "w") as f:
        json.dump(instances_trainval2017_dict, f)

    os.makedirs(f"{train_val_str}", exist_ok=True)
    origin_root = f"{COCO_DATA_ROOT}/{train_val_str}"

    # 拷贝图片
    for img_info in images:
        file_name = img_info['file_name']
        copyfile(f"{origin_root}/{file_name}", f"{train_val_str}/{file_name}")


    # --------------------------- for captions ---------------------------
    captions_file = f"{COCO_DATA_ROOT}/annotations/captions_{train_val_str}.json"
    with open(captions_file) as f:
        captions_trainval2017_dict = json.load(f)
    
    images = captions_trainval2017_dict['images']
    annotations = captions_trainval2017_dict['annotations']

    new_images = []
    for img_info in images:
        if img_info["id"] in img_id_list:
            new_images.append(img_info)

    new_annotations = []
    for ann_info in annotations:
        if ann_info["image_id"] in img_id_list:
            new_annotations.append(ann_info)

    captions_trainval2017_dict['images'] = new_images
    captions_trainval2017_dict['annotations'] = new_annotations

    with open(f"captions_{train_val_str}.json", "w") as f:
        json.dump(captions_trainval2017_dict, f)


    # --------------------------- for person_keypoints ---------------------------
    person_keypoints_file = f"{COCO_DATA_ROOT}/annotations/person_keypoints_{train_val_str}.json"
    with open(person_keypoints_file) as f:
        person_keypoints_trainval2017_dict = json.load(f)
    
    images = person_keypoints_trainval2017_dict['images']
    annotations = person_keypoints_trainval2017_dict['annotations']

    new_images = []
    for img_info in images:
        if img_info["id"] in img_id_list:
            new_images.append(img_info)

    new_annotations = []
    for ann_info in annotations:
        if ann_info["image_id"] in img_id_list:
            new_annotations.append(ann_info)

    person_keypoints_trainval2017_dict['images'] = new_images
    person_keypoints_trainval2017_dict['annotations'] = new_annotations

    with open(f"person_keypoints_{train_val_str}.json", "w") as f:
        json.dump(person_keypoints_trainval2017_dict, f)
    