# -*- coding: utf-8 -*-
# Author: Tao Hu <taohu620@gmail.com>

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from models.ssd.layers.modules import MultiBoxLoss
from tqdm import tqdm
from cl_utils import pytorchgo_logger as logger
from data.cl_datasetv2 import FSLDataset, detection_collate
from cl_utils.ssd_augmentations import SSDAugmentation
from models.ssd.silco_module import build_ssd
import numpy as np
from collections import Counter
from sklearn.metrics import average_precision_score

num_classes = 2
iterations = 40000
stepvalues = (25000,)
log_per_iter = 200
save_per_iter = 5000
shot_num = 5
lr = 1e-4
weight_decay = 5e-4
image_size = 300
batch_size = 4
episode_num = 100
episode_length = 500

parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training"
)
parser.add_argument(
    "--dim",
    default=image_size,
    type=int,
    help="Size of the input image, only support 300 or 512",
)
parser.add_argument(
    "--basenet", default="vgg16_reducedfc.pth", help="pretrained base model"
)
parser.add_argument(
    "--batch_size", default=batch_size, type=int, help="Batch size for training"
)
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help="Number of workers used in dataloading",
)
parser.add_argument(
    "--iterations",
    default=iterations,
    type=int,
    help="Number of training iterations",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=lr,
    type=float,
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument(
    "--weight_decay",
    default=weight_decay,
    type=float,
    help="Weight decay for SGD",
)
parser.add_argument(
    "--subset", default="0", choices=["0", "1", "2", "3"], type=str
)
parser.add_argument(
    "--dataset", default="cl_voc", choices=["cl_voc", "cl_coco14"], type=str
)
parser.add_argument("--shot_num", default=5, type=int, help="shot number")
parser.add_argument("--test", action="store_true")
parser.add_argument("--test_load", default="", type=str)
parser.add_argument("--log_per_iter", default=log_per_iter, type=str)
parser.add_argument("--save_per_iter", default=save_per_iter, type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--visdom", action="store_true")
parser.add_argument("--log_action", default="n", type=str)
parser.add_argument("--variant", default="default", type=str)
parser.add_argument("--episode_num", default=episode_num, type=int)
parser.add_argument("--episode_length", default=episode_length, type=int)

args = parser.parse_args()

customized_logger_dir = "train_log/{}_{}_subset{}_{}".format(
    os.path.basename(__file__).replace(".py", ""),
    args.dataset,
    args.subset,
    args.variant,
)

if args.debug:
    args.iterations = 60
    args.save_per_iter = 20
    args.log_per_iter = 1
    args.episode_num = 2

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

if "coco" in args.dataset:
    args.iterations = args.iterations * 2
    stepvalues = [sv * 2 for sv in stepvalues]
    print("coco dataset! iter num is changed to {}".format(args.iterations))

train_setting = {
    "split": "train",
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}


val_setting = {
    "split": "val",
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}

test_setting = {
    "split": "test",
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}


def adjust_learning_rate(optimizer, lr, iteration):
    if iteration >= stepvalues[0]:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self):
        from visdom import Visdom

        self.viz = Visdom()
        self.plots = {}

    def plot(self, title_name, x, y, xlabel, ylabel):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                opts=dict(title=title_name, xlabel=xlabel, ylabel=ylabel),
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                win=self.plots[title_name],
                update="append",
            )


def train():

    logger.info("current cuda device: {}".format(torch.cuda.current_device()))

    model = build_ssd(args.dim, num_classes)
    vgg16_state_dict = torch.load(args.basenet)
    new_params = {}
    for index, i in enumerate(vgg16_state_dict):
        new_params[i] = vgg16_state_dict[i]
    logger.info("Loading base network...")
    model.query_vgg.load_state_dict(torch.load(args.basenet))
    model = model.cuda()

    def xavier(param):
        init.xavier_uniform_(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    logger.info("Initializing weights...")
    # initialize newly added layers' weights with xavier method
    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)
    model.train()
    criterion = MultiBoxLoss(
        num_classes,
        size=args.dim,
        overlap_thresh=0.5,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=3,
        neg_overlap=0.5,
        encode_target=False,
        use_gpu=True,
    )
    logger.info("Loading Dataset...")
    dataset = FSLDataset(
        params=train_setting,
        image_size=(args.dim, args.dim),
        query_image_augs=SSDAugmentation(args.dim),
        ds_name=args.dataset,
    )
    epoch_size = len(dataset) // args.batch_size

    optimizer = optim.SGD(
        list(model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    data_loader = data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=detection_collate,
    )
    batch_iterator = iter(data_loader)

    best_val_result = 0
    lr = args.lr
    if args.visdom:
        plotter = VisdomLinePlotter()
    for iteration in tqdm(
        range(args.iterations + 1),
        total=args.iterations,
        desc="training {}".format(logger.get_logger_dir()),
    ):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        if iteration in stepvalues:
            lr = adjust_learning_rate(optimizer, lr, iteration)

        # load train data
        support_images, query_image, targets, metadata = next(batch_iterator)
        support_images = Variable(
            support_images.cuda()
        )  # torch.Size([4, 5, 3, 300, 300])
        query_image = Variable(
            query_image.cuda()
        )  # torch.Size([4, 3, 300, 300])
        targets = [Variable(anno.cuda()) for anno in targets]
        out = model(
            support_images=support_images,
            query_image=query_image,
            is_train=True,
        )
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        if iteration % args.log_per_iter == 0 and iteration > 0:
            logger.info(
                """LR: {}\t Iter: {}\t Loss_l: {:.5f}\t Loss_c: {:.5f}\t Loss_total: {:.5f}\t best_result: {:.5f}""".format(
                    lr,
                    iteration,
                    loss_l.item(),
                    loss_c.item(),
                    loss.item(),
                    best_val_result,
                )
            )
            if args.visdom:
                plotter.plot(
                    title_name="train_loss",
                    x=iteration,
                    y=loss.item(),
                    xlabel="iter",
                    ylabel="loss",
                )

        if iteration % args.save_per_iter == 0 and iteration > 0:
            model.eval()
            cur_eval_result = do_eval(
                model, test_setting=val_setting, args=args
            )
            """
            test_result = do_eval(
                model,
                test_setting=test_setting,
                args=args,
            )
            """
            model.train()

            is_best = True if cur_eval_result > best_val_result else False
            if is_best:
                best_val_result = cur_eval_result
                torch.save(
                    {
                        "iteration": iteration,
                        "optim_state_dict": optimizer.state_dict(),
                        "cl_state_dict": model.state_dict(),
                        "best_result": best_val_result,
                    },
                    os.path.join(logger.get_logger_dir(), "best.pth"),
                )
            else:
                logger.info("skip..")

            if args.visdom:
                plotter.plot(
                    title_name="validation_mAP",
                    x=iteration,
                    y=best_val_result,
                    xlabel="iter",
                    ylabel="mAP",
                )
                """
                plotter.plot(
                    title_name="testing_mAP",
                    x=iteration,
                    y=test_result,
                    xlabel="iter",
                    ylabel="mAP",
                )
                """

            logger.info(
                "current iter: {} current_result: {:.5f}".format(
                    iteration, cur_eval_result
                )
            )
            logger.warning("logger dir= {}".format(logger.get_logger_dir()))

    model.eval()
    test_result = do_eval(
        model,
        test_setting=test_setting,
        args=args,
        episode_num=args.episode_num,
    )
    logger.info(
        "test result={:.5f}, best validation result={:.5f}".format(
            test_result, best_val_result
        )
    )
    logger.info(
        "training finish. snapshot weight in {}".format(logger.get_logger_dir())
    )


def bbox_map(detections, groundTruths, classes=[1], IOUThreshold=0.5):
    def cal_bbox_iou(boxA, boxB):
        def _boxesIntersect(boxA, boxB):
            if boxA[0] > boxB[2]:
                return False  # boxA is right of boxB
            if boxB[0] > boxA[2]:
                return False  # boxA is left of boxB
            if boxA[3] < boxB[1]:
                return False  # boxA is above boxB
            if boxA[1] > boxB[3]:
                return False  # boxA is below boxB
            return True

        def _getIntersectionArea(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # intersection area
            return (xB - xA + 1) * (yB - yA + 1)

        def _getUnionAreas(boxA, boxB, interArea=None):
            area_A = _getArea(boxA)
            area_B = _getArea(boxB)
            if interArea is None:
                interArea = _getIntersectionArea(boxA, boxB)
            return float(area_A + area_B - interArea)

        def _getArea(box):
            return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

        # if boxes dont intersect
        if _boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = _getIntersectionArea(boxA, boxB)
        union = _getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    ap_list = []
    for c in classes:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if d["class_id"] == c]
        # Get only ground truths of class c
        gts = []
        [gts.append(g) for g in groundTruths if g["class_id"] == c]
        npos = len(gts)
        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf["score"], reverse=True)
        TP = np.zeros(len(dects))
        TP_score = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = Counter([cc["image_name"] for cc in gts])
        for key, val in det.items():
            det[key] = np.zeros(val)
        # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = [
                gt for gt in gts if gt["image_name"] == dects[d]["image_name"]
            ]
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                iou = cal_bbox_iou(dects[d]["bbox"], gt[j]["bbox"])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if det[dects[d]["image_name"]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    TP_score[d] = dects[d]["score"]
                    det[dects[d]["image_name"]][
                        jmax
                    ] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    TP[d] = 0
                    TP_score[d] = dects[d]["score"]
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                TP[d] = 0
                TP_score[d] = dects[d]["score"]
        ap = average_precision_score(y_true=TP, y_score=TP_score)
        ap_list.append(ap)
    return sum(ap_list) / len(ap_list)


def do_eval(model, test_setting, args, episode_num=1):
    def _do_eval():
        dataset = FSLDataset(
            params=test_setting,
            image_size=(args.dim, args.dim),
            ds_name=args.dataset,
            length=args.episode_length,
        )
        num_images = len(dataset)
        data_loader = data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=detection_collate,
        )

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [
            [[] for _ in range(num_images)] for _ in range(num_classes)
        ]
        w = image_size
        h = image_size

        detection_list = []
        gt_list = []
        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="online {}, episode_num={}".format(
                test_setting["split"], args.episode_num
            ),
        ):
            if args.debug and i > 100:
                break
            support_images, query_image, targets, metadata = batch
            support_images = Variable(support_images.cuda())
            query_image = Variable(query_image.cuda())
            query_origin_img = metadata[0]["cl_query_image"]
            class_name = metadata[0]["class_name"]
            gt_bboxes = targets[0].numpy()
            gt_bboxes[:, 0] = (gt_bboxes[:, 0] * w).astype(int)
            gt_bboxes[:, 1] = (gt_bboxes[:, 1] * h).astype(int)
            gt_bboxes[:, 2] = (gt_bboxes[:, 2] * w).astype(int)
            gt_bboxes[:, 3] = (gt_bboxes[:, 3] * h).astype(int)
            for _ in range(gt_bboxes.shape[0]):
                gt_list.append(
                    dict(
                        image_name=i,
                        class_id=1,
                        bbox=[
                            gt_bboxes[_, 0],
                            gt_bboxes[_, 1],
                            gt_bboxes[_, 2],
                            gt_bboxes[_, 3],
                        ],
                    )
                )
            detections = model(
                support_images=support_images,
                query_image=query_image,
                is_train=False,
            ).data
            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = (
                    dets[:, 0].gt(0).expand(5, dets.size(0)).t()
                )  # greater than 0 will be visualized!
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:].cpu().numpy()
                scores = dets[:, 0].cpu().numpy()
                boxes[:, 0] = (boxes[:, 0] * w).astype(int)
                boxes[:, 1] = (boxes[:, 1] * h).astype(int)
                boxes[:, 2] = (boxes[:, 2] * w).astype(int)
                boxes[:, 3] = (boxes[:, 3] * h).astype(int)

                # padding
                boxes[:, 0][boxes[:, 0] < 0] = 0
                boxes[:, 1][boxes[:, 1] < 0] = 0
                boxes[:, 2][boxes[:, 2] > image_size] = image_size
                boxes[:, 3][boxes[:, 3] > image_size] = image_size

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                    np.float32, copy=False
                )
                all_boxes[j][i] = cls_dets

                for _ in range(cls_dets.shape[0]):
                    detection_list.append(
                        dict(
                            image_name=i,
                            class_id=1,
                            score=cls_dets[_, 4],
                            bbox=[
                                int(cls_dets[_, 0]),
                                int(cls_dets[_, 1]),
                                int(cls_dets[_, 2]),
                                int(cls_dets[_, 3]),
                            ],
                        )
                    )

        mAP = bbox_map(
            detections=detection_list, groundTruths=gt_list, IOUThreshold=0.5
        )
        return mAP

    mAP_list = []
    for _ in range(episode_num):
        mAP_list.append(_do_eval())
    map_np = np.array(mAP_list)
    mAP_mean = map_np.mean()
    mAP_var = map_np.var()
    logger.warning("all episodes result: {}".format(map_np))
    logger.warning(
        "currennt {} mAP={} +/- {}".format(
            test_setting["split"], mAP_mean, mAP_var
        )
    )
    return mAP_mean


if __name__ == "__main__":
    if not args.test:
        logger.set_logger_dir(customized_logger_dir, args.log_action)
        train()
    else:
        print("start test...")
        model = build_ssd(args.dim, num_classes, top_k=200)
        saved_dict = torch.load(args.test_load)
        model.load_state_dict(saved_dict["cl_state_dict"], strict=True)

        model.eval()
        do_eval(
            model=model,
            test_setting=test_setting,
            args=args,
            episode_num=args.episode_num,
        )
        print("online validation result: {}".format(saved_dict["best_result"]))
