import os.path as osp
import sys
from tqdm import tqdm
import os
import numpy as np
import random
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2

VOC_PATH = "/home/tao/dataset/pascalvoc12_07/VOCdevkit"
VOC_SEG_AUG_PATH = (
    "/home/tao/dataset/pascalvoc12_07/VOCdevkit/VOC2012/SegmentationClassAug"
)
augs = [tmp.replace(".png", "") for tmp in os.listdir(VOC_SEG_AUG_PATH)]

val_test_num = 5000
small_size_limit = 0.001
IMG_MEAN_RGB = np.array(
    (122.67891434, 116.66876762, 104.00698793), dtype=np.float32
)
result_json = "cl_voc_v4.json"
MAX_NUM_GT_BOXES = 30

class PASCAL:
    def __init__(self, db_path, dataType):
        assert dataType == "train" or dataType == "val" or dataType == "test"
        self.db_path = db_path
        self.name_id_map = dict(zip(PASCAL_CLASS, range(1, 21)))
        self.id_name_map = dict(zip(range(1, 21), PASCAL_CLASS))
        self.dataType = dataType

        self._annopath = osp.join("%s", "Annotations", "%s.xml")
        self._imgpath = osp.join("%s", "JPEGImages", "%s.jpg")

        self.keep_difficult = False
        self.class_to_ind = dict(
            zip(PASCAL_CLASS, range(1, len(PASCAL_CLASS) + 1))
        )  # start from 1!!!!!

    def getCatIds(self, catNms=[]):
        return [self.name_id_map[catNm] for catNm in catNms]

    def _get_bbox(self, target, width, height):
        res = []
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = (
                    cur_pt * 1.0 / width
                    if i % 2 == 0
                    else cur_pt * 1.0 / height
                )
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]
        return res

    def create_anns(self, catIds=[]):
        tuple_list = []


        rootpath = os.path.join(self.db_path, "VOC2012")
        for line in open(os.path.join("voc12fsd_{}.txt".format(self.dataType))):
            if line.strip() in augs:
                tuple_list.append((rootpath, line.strip()))

        anns = []
        small_size_unqualified = 0
        for item in tqdm(
            tuple_list,
            total=len(tuple_list),
            desc="create annotations  {}".format(self.dataType),
        ):  # per image
            image_root = item[0]
            img_id = item[1]
            class_bbox_dict = {}
            target = ET.parse(self._annopath % item).getroot()
            im = Image.open(self._imgpath % item)
            width, height = im.size

            seg_class_ids = list(
                np.unique(
                    cv2.imread(
                        os.path.join(VOC_SEG_AUG_PATH, "{}.png".format(img_id))
                    )
                )
            )

            target_bboxs = self._get_bbox(target, width, height)

            for bbox in target_bboxs:
                class_id = bbox[-1]
                assert class_id > 0  # no background class
                if class_id not in seg_class_ids:
                    continue
                    print("jump, don't exist corresponding seg label")
                ltrb = bbox[:-1]
                if (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1]) < small_size_limit:
                    small_size_unqualified += 1
                    continue  # skip current
                if class_id in class_bbox_dict:
                    class_bbox_dict[class_id]["bboxes"].append(ltrb)
                else:  # setup new dict
                    d = "VOC2012" if "VOC2012" in image_root else "VOC2007"
                    image_name = "{}/JPEGImages/{}.jpg".format(d, img_id)
                    class_bbox_dict[class_id] = dict(
                        image_name=image_name,
                        class_id=class_id,
                        class_name=PASCAL_CLASS[class_id - 1],
                        bboxes=[ltrb],
                    )

            anns.extend([value for value in class_bbox_dict.values()])

        filtered_anns = [
            ann
            for ann in tqdm(anns, total=len(anns))
            if ann["class_id"] in catIds
        ]  # TODO,filter by object size
        print(
            "filtered/origin anns={}/{}".format(len(filtered_anns), len(anns))
        )
        print(
            "{} images(percentage: {}) are filtered because too small.".format(
                small_size_unqualified,
                small_size_unqualified * 1.0 / len(filtered_anns),
            )
        )
        return filtered_anns

    def getItems(self, cats=[]):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)

        anns_positive = self.create_anns(catIds=catIds)  # heavy operation!

        def process_func(anns_input):
            items = []
            for i, ann in enumerate(anns_input):
                class_id = ann["class_id"]
                bboxes = ann["bboxes"]
                if len(bboxes) >= MAX_NUM_GT_BOXES:
                    print("skip, larger than {}".format(MAX_NUM_GT_BOXES))
                    continue
                items.append(
                    dict(
                        img_path=ann["image_name"],
                        class_id=class_id,
                        bboxes=bboxes,
                    )
                )
            print(
                "data result: total of {} db items loaded!".format(len(items))
            )
            dict_obj2bbox = {}
            for i, item in enumerate(items):
                class_id = item["class_id"]
                if class_id in dict_obj2bbox:
                    dict_obj2bbox[class_id].append(item)
                else:
                    dict_obj2bbox[class_id] = [item]
            return dict_obj2bbox

        return process_func(anns_positive)


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


from data.list.dataset_util import get_cats, PASCAL_CLASS

with open(osp.join(result_json), "w") as f:
    final_dict = {}
    final_dict["train"] = {}
    final_dict["val"] = {}
    final_dict["test"] = {}


    id2cls = dict()
    cls2id = dict()
    for idx, cls_name in enumerate(PASCAL_CLASS):
        id2cls[idx + 1] = cls_name
        cls2id[cls_name] = idx + 1
    final_dict["meta"] = dict(id2cls=id2cls, cls2id=cls2id)

    for dta_type in ["val", "test", "train"]:
        pascal_db = PASCAL(VOC_PATH, dta_type)  # train or test
        final_dict[dta_type] = pascal_db.getItems(cats=PASCAL_CLASS)

    print("dump json file...")
    import json

    json.dump(final_dict, f)
