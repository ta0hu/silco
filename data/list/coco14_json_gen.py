import os.path as osp
import sys
import pickle, json
from tqdm import tqdm
import os
import numpy as np
import random

from pycocotools.coco import COCO
from data.list.dataset_util import (
    coco_cat2trainid,
    get_cats_coco as get_cats,
    coco_num_class,
    coco_trainid2cat,
    COCO14_CLASS,
)


file_name = "cl_coco14"
result_json = "{}_v4.json".format(file_name)


val_test_num = 10000
per_class_min_sample = 0
small_size_limit = 0.1
IMG_MEAN_RGB = np.array(
    (122.67891434, 116.66876762, 104.00698793), dtype=np.float32
)
print("small_size_limit: {}".format(small_size_limit))
MAX_NUM_GT_BOXES = 30

class COCO14:
    def __init__(self, dataType):

        assert dataType == "train" or dataType == "val" or dataType == "test"
        self.name_id_map = dict(zip(COCO14_CLASS, range(1, 81)))
        self.id_name_map = dict(zip(range(1, 81), COCO14_CLASS))
        self.dataType = dataType

        self._anno_train = osp.join("coco_list/instances_train2014.json")
        self._anno_val = osp.join("coco_list/instances_val2014.json")

        self.coco_train = COCO(self._anno_train)
        self.coco_val = COCO(self._anno_val)
        cats = self.coco_train.cats
        self.catid2trainid = {}
        for cat_id, item in cats.items():
            self.catid2trainid[cat_id] = coco_cat2trainid[item["name"]]
        self.fse_split()

        if self.dataType == "train" or self.dataType == "val":
            self.coco = self.coco_train
            if self.dataType == "train":
                self.img_ids = self.train_img_id
            elif self.dataType == "val":
                self.img_ids = self.val_img_id
            else:
                raise
        elif self.dataType == "test":
            self.coco = self.coco_val
            self.img_ids = self.test_img_id
        else:
            raise

    def getCatIds(self, catNms=[]):
        try:
            return [self.name_id_map[catNm] for catNm in catNms]
        except:
            print("a")

    def fse_split(self):
        fse_json = "coco_list/coco14_fse_train_val_test.json"
        if os.path.exists(fse_json):
            tmp = json.load(open(fse_json, "r"))
            self.train_img_id = tmp["train"]
            self.val_img_id = tmp["val"]
            self.test_img_id = tmp["test"]
        else:
            val_num = 20000
            train_num = 62783
            test_num = 40504
            with open(fse_json, "w") as f:
                trainval = list(self.coco_train.imgs.keys())
                random.shuffle(trainval)
                train = trainval[:train_num]
                val = trainval[train_num:]
                test = list(self.coco_val.imgs.keys())
                json.dump(dict(train=train, val=val, test=test), f)

                self.train_img_id = train
                self.val_img_id = val
                self.test_img_id = test

    def create_anns(self, catIds=[]):
        def bbox_coco_to_voc(bbox, height, width):
            l = bbox[0]
            t = bbox[1]
            w = bbox[2]
            h = bbox[3]
            return [
                l * 1.0 / width,
                t * 1.0 / height,
                (l + w) * 1.0 / width,
                (t + h) * 1.0 / height,
            ]

        anns = []
        small_size_unqualified = 0
        for img_id in tqdm(
            self.img_ids,
            total=len(self.img_ids),
            desc="create annotations {}".format(self.dataType),
        ):  # per image
            _img = self.coco.loadImgs(ids=[img_id])[0]
            height = _img["height"]
            width = _img["width"]
            file_name = _img["file_name"]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            class_bbox_dict = {}
            if "train" in file_name:
                file_name = "train2014/{}".format(file_name)
            elif "val" in file_name:
                file_name = "val2014/{}".format(file_name)
            else:
                raise

            for ann in self.coco.loadAnns(ids=ann_ids):
                bbox = ann["bbox"]
                class_train_id = self.catid2trainid[ann["category_id"]]
                if (
                    (bbox[2] * bbox[3] * 1.0) / (height * width)
                ) < small_size_limit:
                    small_size_unqualified += 1
                    continue  # skip current
                bbox = bbox_coco_to_voc(bbox, height=height, width=width)
                if class_train_id in class_bbox_dict:
                    class_bbox_dict[class_train_id]["bboxes"].append(bbox)
                    class_bbox_dict[class_train_id]["segmentations"].append(
                        ann["segmentation"]
                    )
                else:  # setup new dict
                    class_bbox_dict[class_train_id] = dict(
                        image_name=file_name,
                        class_id=class_train_id,
                        class_name=coco_trainid2cat[class_train_id],
                        coco_category_id=ann["category_id"],
                        coco_image_id=img_id,
                        bboxes=[bbox],
                        height=height,
                        width=width,
                        segmentations=[ann["segmentation"]],
                    )

            anns.extend([value for value in class_bbox_dict.values()])

        filtered_anns = [
            ann
            for ann in tqdm(anns, total=len(anns))
            if ann["class_id"] in catIds
        ]
        print(
            "filtered/origin anns={}/{}".format(len(filtered_anns), len(anns))
        )
        print(
            "{} images(percentage: {}) are filtered because too small.".format(
                small_size_unqualified,
                small_size_unqualified * 1.0 / len(filtered_anns),
            )
        )

        # clustering
        items = []
        for i, ann in enumerate(filtered_anns):  # TODO, dunplicate function
            item = dict(
                img_path=ann["image_name"],
                class_id=ann["class_id"],
                coco_category_id=ann["coco_category_id"],
                coco_image_id=ann["coco_image_id"],
                class_name=self.id_name_map[ann["class_id"]],
                bboxes=ann["bboxes"],
                height=ann["height"],
                width=ann["width"],
            )

            if len(item['bboxes']) >= MAX_NUM_GT_BOXES:
                print("skip, larger than {}".format(MAX_NUM_GT_BOXES))
                continue

            items.append(item)

        print("data result: total of {} db items loaded!".format(len(items)))
        clusters = {}
        for i, item in enumerate(items):
            item_id = item["class_id"]
            if item_id in clusters:
                clusters[item_id].append(item)
            else:
                clusters[item_id] = [item]
        new_cluster = {}

        for key, value in clusters.items():
            if len(value) > per_class_min_sample:
                new_cluster[key] = value
                print("keep in class {}, len={}".format(key, len(value)))
            else:
                print(
                    "******filter out class {}, len={}".format(key, len(value))
                )
            """
            for item in value:
                annIds = self.coco.getAnnIds(imgIds=[item['coco_image_id']])
                anns = self.coco.loadAnns(annIds)
                cat_set = set()
                for ann in anns:
                    m = self.coco.annToMask(ann)
                    cat_set.add(self.catid2trainid[ann['category_id']])
                if key not in cat_set:
                    print("a")
                assert key in cat_set
            """

        return new_cluster

    def getItems(self, cats=[]):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)
        anns_positive = self.create_anns(catIds=catIds)
        return anns_positive


def generate_limited_samples(positive_imgsets, cls_id_list):
    def generate_tuple(positive_imgsets, max_shot=10):
        chosen_class_id = random.choice(cls_id_list)
        positive_imgset = positive_imgsets[chosen_class_id]

        _index = list(range(len(positive_imgset)))
        random.Random().shuffle(_index)
        _index = _index[: (max_shot + 1)]
        support_index = _index[:max_shot]
        query_index = _index[-1]
        return support_index, query_index, chosen_class_id

    proceeded_item = []
    # different strategy for val/test and train data, val/test pair number=3000
    # if postive, query can be target class, support can also be target class
    # if negative, query can not be target class, support can also be target class

    for iii in range(val_test_num):
        support_index, query_index, chosen_class_id = generate_tuple(
            positive_imgsets
        )
        proceeded_item.append(
            dict(
                class_id=chosen_class_id,
                query_index=query_index,
                support_index=support_index,
            )
        )
    return proceeded_item


with open(osp.join(result_json), "w") as f:
    final_dict = {}
    final_dict["train"] = {}
    final_dict["val"] = {}
    final_dict["test"] = {}
    final_dict["meta"] = dict(id2cls=coco_trainid2cat, cls2id=coco_cat2trainid)

    for dta_type in ["val", "test", "train"]:
        print("generating {}".format(dta_type))
        coco14_db = COCO14(dta_type)  # train or test
        final_dict[dta_type] = coco14_db.getItems(COCO14_CLASS)


    print("dump json file...")
    import json

    json.dump(final_dict, f)
