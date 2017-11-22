import os
import os.path
import sys
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import cv2
import numpy as np

cats_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
           46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
           67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

id_to_idx = dict(zip(cats_id, range(len(cats_id))))
idx_to_id = dict(zip(range(len(cats_id)), cats_id))

class COCODetection(data.Dataset):
    """COCO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to COCO folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=True,
                 dataset_name='COCO'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(self.root, 'annotations', '%s.json')
        self._imgpath = os.path.join('%s',  '%s')
        self.ids = list()
        self.test = list()
        self.imgs = dict()
        self.annos = defaultdict(list)


        root_path = os.path.join(self.root, image_sets)
        with open(self._annopath%image_sets, 'r') as f:
            dataset = json.load(f)

        for anno in dataset['annotations']:
            self.annos[anno['image_id']].append(anno)

        for img in dataset['images']:
            self.imgs[img['id']] = img
            #self.ids.append((root_path, img['id']))

        self.ids = [(root_path, img_id) for img_id in self.annos.keys()]
               
        
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        root_path, img_id = self.ids[index]
        anno = self.annos[img_id]
        img  = self.imgs[img_id]

        img_data = cv2.imread(self._imgpath % (root_path, img['file_name']))
        bboxes = None
        labels = None
        if self.target_transform:
            bboxes, labels = self.anno_transform(anno, img['height'], img['width'], img_id)

        if self.transform is not None:
            img_data, bboxes, labels = self.transform(img_data, bboxes, labels)
            # to rgb
            img_data = img_data[:, :, (2, 1, 0)]
            target = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img_data).permute(2, 0, 1), target, img['height'], img['width']


    def anno_transform(self, anno, height, width, img_id):

        bboxes = []
        labels = []
        for item in anno:
            bboxes.append(item['bbox'])
            labels.append(id_to_idx[item['category_id']])

        bboxes = np.array(bboxes)
        labels = np.array(labels)

        bboxes[:, 0::2] = bboxes[:, 0::2] / width
        bboxes[:, 1::2] = bboxes[:, 1::2] / height
        # convert (x_min, y_min, w, h) to (x_min, y_min, x_max, y_max)
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]

        return bboxes, labels

    def pull_image(self, index):
        
        root_path, img_id = self.ids[index]
        img = self.imgs[img_id]

        return cv2.imread(self._imgpath % (root_path, img['file_name']), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        
        root_path, img_id = self.ids[index]
        anno = self.annos[img_id]
        img = self.imgs[img_id]

        bboxes, labels = self.anno_transform(anno, img['height'], img['width'])
        target = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        return img_id, target

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


