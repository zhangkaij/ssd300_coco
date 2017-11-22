"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import COCOroot, idx_to_id
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import COCODetection, BaseTransform
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--root_path', default=COCOroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)
set_type = 'test'
ssd_dim = 300
num_classes = 81
img_sets = "val2017"

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(net, cuda, dataset):
    """Test a SSD network on an image database."""
    num_images = len(dataset)
    
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    

    sum_time = 0.0
    sum_fps  = 0.0
    output = list()
    anno = dict()

    for i in range(num_images):
        _, img_id = dataset.ids[i]
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        sum_time += detect_time
        sum_fps += 1.0 / detect_time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # convert (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            scores = dets[:, 0].cpu().numpy()
            boxes = boxes.cpu().numpy()

            for box in boxes:
                anno["image_id"] = img_id
                anno["category_id"] = idx_to_id[j]
                anno["bbox"] = list(box)
                anno["score"] = scores
                output.append(anno)
            

        #print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
        #                                            num_images, detect_time))

    print('time: ', sum_time/num_images, 'fps: ', sum_fps/num_images)
    print('sum: ', net.detect.count, 'mean: ', net.detect.count/num_images)
    
    print('writing detections')
    output_path = os.path.join(args.root_path, 'result/result.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)
    

if __name__ == '__main__':
    # load net
    net = build_ssd('test', ssd_dim, num_classes)    # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = COCODetection(args.root_path, img_sets, BaseTransform(ssd_dim, dataset_mean), target_transform=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net, args.cuda, dataset)
