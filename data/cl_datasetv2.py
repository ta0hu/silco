from cl_utils import pytorchgo_logger as logger
import os
import numpy as np
import cv2
import torch, random
import torch.utils.data as data
from data.rect import draw_det_box_seg
from PIL import Image
from pycocotools.coco import COCO

coco_cat2trainid = {
    "toilet": 62,
    "teddy bear": 78,
    "cup": 42,
    "bicycle": 2,
    "kite": 34,
    "carrot": 52,
    "stop sign": 12,
    "tennis racket": 39,
    "donut": 55,
    "snowboard": 32,
    "sandwich": 49,
    "motorcycle": 4,
    "oven": 70,
    "keyboard": 67,
    "scissors": 77,
    "airplane": 5,
    "couch": 58,
    "mouse": 65,
    "fire hydrant": 11,
    "boat": 9,
    "apple": 48,
    "sheep": 19,
    "horse": 18,
    "banana": 47,
    "baseball glove": 36,
    "tv": 63,
    "traffic light": 10,
    "chair": 57,
    "bowl": 46,
    "microwave": 69,
    "bench": 14,
    "book": 74,
    "elephant": 21,
    "orange": 50,
    "tie": 28,
    "clock": 75,
    "bird": 15,
    "knife": 44,
    "pizza": 54,
    "fork": 43,
    "hair drier": 79,
    "frisbee": 30,
    "umbrella": 26,
    "bottle": 40,
    "bus": 6,
    "bear": 22,
    "vase": 76,
    "toothbrush": 80,
    "spoon": 45,
    "train": 7,
    "sink": 72,
    "potted plant": 59,
    "handbag": 27,
    "cell phone": 68,
    "toaster": 71,
    "broccoli": 51,
    "refrigerator": 73,
    "laptop": 64,
    "remote": 66,
    "surfboard": 38,
    "cow": 20,
    "dining table": 61,
    "hot dog": 53,
    "car": 3,
    "sports ball": 33,
    "skateboard": 37,
    "dog": 17,
    "bed": 60,
    "cat": 16,
    "person": 1,
    "skis": 31,
    "giraffe": 24,
    "truck": 8,
    "parking meter": 13,
    "suitcase": 29,
    "cake": 56,
    "wine glass": 41,
    "baseball bat": 35,
    "backpack": 25,
    "zebra": 23,
}


coco_trainid2cat = {value: key for key, value in coco_cat2trainid.items()}
COCO14_CLASS = [coco_trainid2cat[i] for i in range(1, 81)]
PASCAL_CLASS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
IMG_MEAN_RGB = np.array(
    (122.67891434, 116.66876762, 104.00698793), dtype=np.float32
)

coco14_json_path = "data/list/cl_coco14_v4.json"
voc_json_path = "data/list/cl_voc_v4.json"
import getpass

VOC_PATH = "/home/tao/dataset/pascalvoc12_07/VOCdevkit".replace(
    "tao", getpass.getuser()
)
COCO_PATH = "/home/tao/dataset/coco14".replace("tao", getpass.getuser())
COCO_SEGLABEL_PATH = "/home/tao/dataset/coco14/trainval2014_seg_label".replace(
    "tao", getpass.getuser()
)


IS_DEBUG = 0
if IS_DEBUG == 1:
    print("IS_Debug is true!")


def apply_mask(image, mask, color=np.array([0, 255, 255]), alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c],
        )
    return image


def generate_tuple(_imgsets, shot_num, ds_name, subset_id, trainval):
    if "voc" in ds_name:
        from data.list.dataset_util import get_cats
    elif "coco" in ds_name:
        from data.list.dataset_util import get_cats_coco as get_cats
    else:
        raise
    train_class_dicts = get_cats(subset_id)[trainval]
    train_class_ids = [int(i["cls_id"]) for i in train_class_dicts]

    chosen_class_id = str(random.choice(train_class_ids))
    _imgset = _imgsets[chosen_class_id]

    _index = list(range(len(_imgset)))
    random.Random().shuffle(_index)
    _index = _index[: (shot_num + 1)]
    support_index = _index[:shot_num]
    query_index = _index[-1]

    return support_index, query_index, chosen_class_id


class FSLDataset(data.Dataset):
    def __init__(
        self,
        params,
        image_size=(300, 300),
        ds_name="cl_voc",
        query_image_augs=None,
        length=5000,
    ):
        self.image_size = image_size
        self.params = params
        self.ds_name = ds_name

        def generate_id2trainid(_coco):
            from data.cl_datasetv2 import coco_cat2trainid

            catid2trainid = {}
            for idx, data in enumerate(_coco.cats.items()):
                category_id, category_dict = data  # start from 1!!!
                category_name = category_dict["name"]
                catid2trainid[category_id] = coco_cat2trainid[category_name]
            return catid2trainid

        if self.ds_name == "cl_voc":
            self._CLASS = PASCAL_CLASS
            self.json_path = voc_json_path
        elif self.ds_name == "cl_coco14":
            self._CLASS = COCO14_CLASS
            self.json_path = coco14_json_path
            if "train" in self.params["split"] or "val" in self.params["split"]:
                self._coco = COCO(
                    "{}/annotations/instances_train2014.json".format(COCO_PATH)
                )
                self.catid2trainid = generate_id2trainid(self._coco)
            elif "test" in self.params["split"]:
                self._coco = COCO(
                    "{}/annotations/instances_val2014.json".format(COCO_PATH)
                )
                self.catid2trainid = generate_id2trainid(self._coco)
            else:
                raise
        else:
            raise

        self.load_items()


        self.data_size = (
                length if "voc" in self.ds_name else length * 4
            )  # coco is four times of pascal voc
        self.data_split = self.data_json[self.params["split"]]
        self.query_image_augs = query_image_augs

    def __len__(self):
        return self.data_size

    def load_items(self):
        with open(self.json_path, "r") as f:
            import json

            logger.warn(
                "loading  data from json file {}....".format(self.json_path)
            )
            self.data_json = json.load(f)

    def __getitem__(self, index):
        def get_item(idx):
            support_index, query_index, class_train_id = generate_tuple(
                self.data_split,
                self.params["k_shot"],
                ds_name=self.ds_name,
                subset_id=self.params["subset_id"],
                trainval=self.params["split"],
            )
            support_imgset = self.data_split[class_train_id]
            query_imgset = self.data_split[class_train_id]

            def get_img_path(file_name):
                if "coco" in self.ds_name:
                    return os.path.join(COCO_PATH, file_name)
                elif "voc" in self.ds_name:
                    return os.path.join(VOC_PATH, file_name)  # VOCdevkit
                else:
                    raise

            if "voc" in self.ds_name:

                def generate_voc_mask(img_path):
                    segclass_dir = (
                        "SegmentationClassAug"
                        if "VOC2012" in img_path
                        else "SegmentationClass"
                    )
                    mask_path = os.path.join(
                        VOC_PATH,
                        img_path.replace("JPEGImages", segclass_dir).replace(
                            ".jpg", ".png"
                        ),
                    )
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    assert mask is not None
                    final_mask = np.zeros_like(mask)
                    final_mask[np.where(mask == int(class_train_id))] = 1
                    if np.sum(final_mask) == 0:
                        print("bug")
                    assert np.sum(final_mask) > 0
                    final_mask = cv2.resize(
                        final_mask, self.image_size, cv2.INTER_NEAREST
                    )
                    return final_mask

                segmentation_supports = [
                    generate_voc_mask(support_imgset[_]["img_path"])
                    for _ in support_index
                ]
                segmentation_query = generate_voc_mask(
                    query_imgset[query_index]["img_path"]
                )
            elif "coco" in self.ds_name:

                def generate_coco_mask(coco_img_id):
                    img = self._coco.loadImgs(coco_img_id)[0]
                    annIds = self._coco.getAnnIds(imgIds=[coco_img_id])
                    anns = self._coco.loadAnns(annIds)
                    anns = [
                        ann
                        for ann in anns
                        if self.catid2trainid[ann["category_id"]]
                        == int(class_train_id)
                    ]
                    img_mask = np.zeros(
                        (img["height"], img["width"]), dtype=np.uint8
                    )
                    for ann in anns:
                        m = self._coco.annToMask(ann)
                        img_mask[np.where(m == 1)] = str(class_train_id)
                    img_mask = img_mask
                    final_mask = np.zeros_like(img_mask)
                    final_mask[img_mask == int(class_train_id)] = 1
                    assert np.sum(final_mask) > 0
                    final_mask = cv2.resize(
                        final_mask, self.image_size, cv2.INTER_NEAREST
                    )
                    return final_mask

                segmentation_supports = [
                    generate_coco_mask(support_imgset[_]["coco_image_id"])
                    for _ in support_index
                ]
                segmentation_query = generate_coco_mask(
                    query_imgset[query_index]["coco_image_id"]
                )
            else:
                raise

            metadata = dict(
                class_id=class_train_id,
                class_name=self._CLASS[int(class_train_id) - 1],
                query_image_path=get_img_path(
                    query_imgset[query_index]["img_path"]
                ),
                segmentation_query=segmentation_query,
                segmentation_supports=segmentation_supports,
            )

            return (
                [
                    get_img_path(support_imgset[v]["img_path"])
                    for v in support_index
                ],
                [support_imgset[v]["bboxes"] for v in support_index],
                get_img_path(query_imgset[query_index]["img_path"]),
                query_imgset[query_index]["bboxes"],
                metadata,
            )

        def read_BGR_PIL_and_resize(img_path):
            result_image = np.asarray(
                Image.open(img_path).convert("RGB"), dtype=np.float32
            )
            result_image = cv2.resize(result_image, self.image_size)[
                :, :, [2, 1, 0]
            ]
            return (
                result_image
            )  # RGB to BGR for later ass opencv imwrite of network feed!

        support_images, support_bboxs, query_image, query_bbox, metadata = get_item(
            index
        )

        query_bboxs_original = np.stack(query_bbox, axis=0)
        query_image = read_BGR_PIL_and_resize(query_image)
        origin_query_image = np.copy(query_image).astype(np.uint8)

        k_shot = len(support_images)
        origin_support_images = []
        output_support_images_concat = []

        for k in range(k_shot):
            support_image = support_images[k]
            support_image = read_BGR_PIL_and_resize(support_image)
            origin_support_image = np.copy(support_image)
            bboxs = support_bboxs[k]

            origin_support_image = draw_det_box_seg(
                origin_support_image,
                bboxs,
                class_name=metadata["class_name"],
                seg_mask=metadata["segmentation_supports"][k],
                color=(255, 0, 0),
            )
            origin_support_images.append(origin_support_image)

            masked = np.copy(support_image)
            masked -= IMG_MEAN_RGB  # sub BGR mean
            masked = masked.transpose((2, 0, 1))  # W,H,C->C,W,H
            output_support_images_concat.append(masked)

        origin_query_image = draw_det_box_seg(
            origin_query_image,
            query_bboxs_original,
            color=(255, 0, 0),
            seg_mask=metadata["segmentation_query"],
            class_name=metadata["class_name"],
        )
        if False:
            to_be_drawed = [
                support_img[:, :, [2, 1, 0]]
                for support_img in origin_support_images
            ] + [
                origin_query_image[:, :, [2, 1, 0]]
            ]  # [bgr->rgb]
            im = Image.fromarray(np.uint8(np.concatenate(to_be_drawed, axis=1)))
            # im.save("{}/{}_query_image_{}.jpg".format(logger.get_logger_dir(),index, metadata["class_name"]))
            im.save(
                "{}/{}_query_image_{}.jpg".format(
                    logger.get_logger_dir(), index, metadata["class_name"]
                )
            )
            print("class_name: {}".format(metadata["class_name"]))

        for i, bb in enumerate(query_bbox):
            bb.append(1)  # add default class, notice here!!!

        if self.query_image_augs is not None:  # query_image_augs is necessary!
            query_bbox_np = np.stack(query_bbox, axis=0)
            img, boxes, labels = self.query_image_augs(
                query_image, query_bbox_np[:, :4], query_bbox_np[:, 4]
            )
            img -= IMG_MEAN_RGB
            query_image = img.transpose((2, 0, 1))  # W,H,C->C,W,H
            query_bbox = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            assert query_bbox.shape[1] == 5
        else:
            # for segmentation, no augmentation is conducted
            query_image -= IMG_MEAN_RGB
            query_image = query_image.transpose((2, 0, 1))  # W,H,C->C,W,H

        output_support_images_concat = np.stack(
            output_support_images_concat, axis=0
        )
        output_support_images_concat = np.squeeze(
            output_support_images_concat
        )  # only for one-shot!!!

        metadata["cl_query_image"] = origin_query_image
        metadata["cl_support_images"] = origin_support_images

        return (
            torch.from_numpy(output_support_images_concat.copy()),
            torch.from_numpy(query_image.copy()),
            torch.FloatTensor(query_bbox),
            metadata,
        )


def detection_collate(batch):
    support_images = []
    query_images = []
    query_bboxes = []
    metadata_list = []
    for sample in batch:
        support_images.append(sample[0])
        query_images.append(sample[1])
        query_bboxes.append(sample[2])
        metadata_list.append(sample[3])
    return (
        torch.stack(support_images, 0),
        torch.stack(query_images, 0),
        query_bboxes,
        metadata_list,
    )


if __name__ == "__main__":
    params = {"split": "test", "k_shot": 3, "subset_id": "1"}
    dataset = FSLDataset(params, query_image_augs=None)
    data_loader = data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=detection_collate,
    )

    for idx, data in enumerate(data_loader):
        print(idx)
