# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
import os, cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask

# this file is used to generate segmentation label file for coco, for saving image loading times.
coco_img_path = "/home/tao/dataset/coco14/train2014"
coco_seglabel_path = "/home/tao/dataset/coco14/trainval2014_seg_label"
_coco_train = COCO(
    "/home/tao/dataset/coco14/annotations/instances_train2014.json"
)
_coco_val = COCO("/home/tao/dataset/coco14/annotations/instances_val2014.json")


if not os.path.exists(coco_seglabel_path):
    os.mkdir(coco_seglabel_path)


def generate_mask(_coco, catid2trainid, img_id):
    img = _coco.loadImgs(img_id)[0]
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img["height"], img["width"], 1), dtype=np.uint8)
    for annId in annIds:
        ann = _coco.loadAnns(annId)[0]
        m = _coco.annToMask(ann)
        img_mask[m == 1] = catid2trainid[ann["category_id"]]
    return img_mask


def generate_id2trainid(_coco):
    from data.cl_datasetv2 import coco_cat2trainid

    catid2trainid = {}
    for idx, data in enumerate(_coco.cats.items()):
        category_id, category_dict = data  # start from 1!!!
        category_name = category_dict["name"]
        catid2trainid[category_id] = coco_cat2trainid[category_name]
    return catid2trainid


def save_img(_coco=_coco_train):
    catid2trainid = generate_id2trainid(_coco)
    print(catid2trainid)
    for key, value in tqdm(_coco.imgs.items()):
        img_id = value["id"]
        if img_id == 72821:
            print("a")
        img_name = value["file_name"]
        img_mask = generate_mask(_coco, catid2trainid, img_id=img_id)
        # cv2.imwrite(os.path.join(coco_seglabel_path, img_name), img_mask)
    print("catId_to_ascendorder length: {}".format(len(catid2trainid)))


save_img(_coco_train)
save_img(_coco_val)
