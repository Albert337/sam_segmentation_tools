#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def load_model(device="cpu"):
    sam_checkpoint = "../weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def model_seg1(sam,image):

    mask_generator = SamAutomaticMaskGenerator(sam)


    masks = mask_generator.generate(image)


    # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
    # * `segmentation` : the mask
    # * `area` : the area of the mask in pixels
    # * `bbox` : the boundary box of the mask in XYWH format
    # * `predicted_iou` : the model's own prediction for the quality of the mask
    # * `point_coords` : the sampled input point that generated this mask
    # * `stability_score` : an additional measure of mask quality
    # * `crop_box` : the crop of the image used to generate this mask in XYWH format

    print(len(masks))
    print(masks[0].keys())


    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    return masks

def model_seg2(sam,image):
    # ## Automatic mask generation options

    # There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. Here is an example configuration that samples more masks:

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )



    masks2 = mask_generator_2.generate(image)

    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks2)
    # plt.axis('off')
    # plt.show()
    return masks2

def main():
    # import pdb;pdb.set_trace()
    image = cv2.imread('../notebooks/images/8000.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model=load_model()
    # model_seg1(model,img)
    model_seg2(model,img)

if __name__ == '__main__':
    main()