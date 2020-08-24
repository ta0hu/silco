# Author: Tao Hu <taohu620@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.ssd.layers import *
from data import v as cfg
import os
from cl_utils import pytorchgo_logger as logger


class SSD(nn.Module):
    def __init__(
        self,
        size,
        base,
        extras,
        head,
        num_classes,
        top_k=200,
        conf_thresh=0.01,
        nms_thresh=0.45,
        gcn_layer=5,
    ):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(cfg[str(size)])
        self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.query_vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        logger.info("detection top_k: {}".format(top_k))
        logger.info("detection conf_thresh: {}".format(conf_thresh))
        logger.info("detection nms_thresh: {}".format(nms_thresh))

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(
            num_classes,
            self.size,
            bkg_label=0,
            top_k=top_k,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )

    def forward(self, support_images, query_image, is_train):
        x = torch.cat((support_images, query_image.unsqueeze(dim=1)), 1)
        batch_size, image_num, c, w, h = x.shape
        x = x.view(-1, c, w, h)

        sources = list()
        loc = list()
        conf = list()
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.query_vgg[k](x)

        s = self.L2Norm(x)

        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.query_vgg)):
            if k == 30:
                x = self.query_vgg[k](x)
            else:
                x = self.query_vgg[k](x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        def fusion(source_feat):
            feat_query = []
            feat_support = []
            for _ in range(len(source_feat)):
                cur_scale = source_feat[_]
                cur_scale = cur_scale.view(
                    batch_size,
                    image_num,
                    cur_scale.shape[1],
                    cur_scale.shape[2],
                    cur_scale.shape[3],
                )
                tmp = cur_scale[:, :5, :, :, :].contiguous()
                feat_support.append(
                    tmp.view(-1, tmp.shape[2], tmp.shape[3], tmp.shape[4])
                )
                feat_query.append(cur_scale[:, 5, :, :, :])

            ms_feat_list = []
            for context_level in range(len(feat_query)):
                _feat_query = feat_query[context_level]
                # _feat_support = feat_support[context_level]
                fused_feat = _feat_query
                ms_feat_list.append(fused_feat)
            return ms_feat_list

        sources = fusion(sources)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(
                l(x).permute(0, 2, 3, 1).contiguous()
            )  # BCWH->BWHC,[8,38,38,4*4],[8,19,19,4*6],...
            conf.append(
                c(x).permute(0, 2, 3, 1).contiguous()
            )  # BCWH->BWHC[8,38,38,2*4],[8,19,19,2*6],...

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # B,W*H*C
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if not is_train:  # not is_train.data.numpy()[0]:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(
                    conf.view(conf.size(0), -1, self.num_classes)
                ),  # conf preds
                self.priors.type(type(x.data)),  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage)
            )
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5,
        conv6,
        nn.ReLU(inplace=True),
        conv7,
        nn.ReLU(inplace=True),
    ]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1,
                    )
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # SSD512 need add one more Conv layer(Conv12_2)
    if size == 512:
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        try:
            loc_layers += [
                nn.Conv2d(
                    vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1
                )
            ]
            conf_layers += [
                nn.Conv2d(
                    vgg[v].out_channels,
                    cfg[k] * num_classes,
                    kernel_size=3,
                    padding=1,
                )
            ]
        except:
            import ipdb

            ipdb.set_trace()
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(
                v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1
            )
        ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    "300": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "512": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
}
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
}
mbox = {
    "300": [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    "512": [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(
    size=512, num_classes=21, top_k=200, conf_thresh=0, nms_thresh=0.45
):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 or SSD512 is supported currently!")
        return

    base_, extras_, head_ = multibox(
        vgg(base[str(size)], 3),
        add_extras(extras[str(size)], size, 1024),
        mbox[str(size)],
        num_classes,
    )

    return SSD(
        size,
        base_,
        extras_,
        head_,
        num_classes,
        top_k=top_k,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
    )
