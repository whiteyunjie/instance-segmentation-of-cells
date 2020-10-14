from __future__ import absolute_import

import cv2
import imageio
import numpy as np
import os
import os.path as osp
    
def visual(img, gt):
    img = cv2.imread(img, -1)
    gt = cv2.imread(gt, -1)
    label = np.unique(gt)
    height, width = img.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[gt==lab, :] = color
    return img.astype(np.uint8), visual_img.astype(np.uint8)

def visualimg(img):
    img = cv2.imread(img, -1)
    return img.astype(np.uint8)

if __name__ == "__main__":
    # dataset1
    '''
    image_path = './dataset1/train'
    gt_path = './dataset1/train_GT/SEG'
    images = sorted([osp.join(image_path, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
    gts = sorted([osp.join(gt_path, gt) for gt in os.listdir(gt_path) if gt.find('.tif') != -1])
    visual_path = './dataset1/visual'
    if not osp.exists(visual_path):
        os.mkdir(visual_path)
    for idx, (image, gt) in enumerate(zip(images, gts)):
        img, visual_img = visual(image, gt)
        cv2.imwrite(osp.join(visual_path, '{:0>3d}_visual.jpg'.format(idx)), visual_img.astype(np.uint8))
        cv2.imwrite(osp.join(visual_path, '{:0>3d}_origin_img.jpg'.format(idx)), img.astype(np.uint8))
    '''
   
    # dataset2
    image_path = './dataset2/train'
    gt_path = './dataset2/train_GT/SEG'
    images = sorted([osp.join(image_path, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
    gts = sorted([osp.join(gt_path, gt) for gt in os.listdir(gt_path) if gt.find('.tif') != -1])[-8:]
    visual_path = './dataset2/visual2'
    if not osp.exists(visual_path):
        os.mkdir(visual_path)
    count = 0
    for idx, image in enumerate(images):
        img = visualimg(image)
        #cv2.imwrite(osp.join(visual_path, '{:0>3d}_visual.jpg'.format(count)), visual_img.astype(np.uint8))
        cv2.imwrite(osp.join(visual_path, '{:0>3d}_origin_img.jpg'.format(count)), img.astype(np.uint8))
        count += 1
   