from __future__ import absolute_import

import cv2
import imageio
import numpy as np
import os
import os.path as osp

def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))

def img_standardization(img):
    img = unit16b2uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')
    
def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        img = img_standardization(img)
        images.append(img)
    return images


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 


class BinaryThresholding:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        gray = bgr_to_gray(img)
        (_, binary_mask) = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.medianBlur(binary_mask, 5)
        connectivity = 4
        _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask , connectivity , cv2.CV_32S)
        return label_img
    

if __name__ == "__main__":
    segmentor = BinaryThresholding(threshold=110)
    image_path = './dataset1/test/'
    result_path = './dataset1/test_RES'
    if not osp.exists(result_path):
        os.mkdir(result_path)
    image_list = sorted([osp.join(image_path, image) for image in os.listdir(image_path)])
    images = load_images(image_list)
    for index, image in enumerate(images):
        label_img = segmentor(image)
        imageio.imwrite(osp.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))