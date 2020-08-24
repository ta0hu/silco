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

coco_num_class = 81
coco_trainid2cat = {value: key for key, value in coco_cat2trainid.items()}
COCO14_CLASS = [coco_trainid2cat[i] for i in range(1, coco_num_class)]


def get_cats(fold, num_folds=4):
    fold = int(fold)
    total_class_num = len(PASCAL_CLASS)
    assert total_class_num % num_folds == 0
    assert fold < num_folds

    val_size = total_class_num // num_folds
    valtest_set = [fold * val_size + v for v in range(val_size)]
    train_set = [x for x in range(total_class_num) if x not in valtest_set]

    return dict(
        train=[
            dict(cls_name=PASCAL_CLASS[x], cls_id=x + 1) for x in train_set
        ],  # start from 1
        val=[
            dict(cls_name=PASCAL_CLASS[x], cls_id=x + 1) for x in valtest_set
        ],  # start from 1
        test=[
            dict(cls_name=PASCAL_CLASS[x], cls_id=x + 1) for x in valtest_set
        ],  # start from 1
    )


def get_cats_coco(fold, num_folds=4):
    fold = int(fold)
    total_class_num = len(coco_cat2trainid.keys())
    assert total_class_num % num_folds == 0
    assert fold < num_folds

    val_size = total_class_num // num_folds
    valtest_set = [fold * val_size + v for v in range(val_size)]
    train_set = [x for x in range(total_class_num) if x not in valtest_set]

    return dict(
        train=[
            dict(cls_name=coco_trainid2cat[x + 1], cls_id=x + 1)
            for x in train_set
        ],  # start from 1
        val=[
            dict(cls_name=coco_trainid2cat[x + 1], cls_id=x + 1)
            for x in valtest_set
        ],  # start from 1
        test=[
            dict(cls_name=coco_trainid2cat[x + 1], cls_id=x + 1)
            for x in valtest_set
        ],  # start from 1
    )
