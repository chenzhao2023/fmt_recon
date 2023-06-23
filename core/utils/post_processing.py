import pandas as pd
import re
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings("ignore", category=DeprecationWarning)
from glob import glob

import os
import cv2
import imageio
import matplotlib.pyplot as plt

from skimage.measure import label
from scipy import ndimage
from tqdm import tqdm

plt.rcParams["figure.figsize"]=20,20
from core.utils.evaluate import evaluate_single_image, threshold_by_otsu, dice_coefficient_in_train, misc_measures_in_train


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def post_processing_results_dual(bin, surface, radius):
    is_surface_frames = np.array([np.count_nonzero(surface[:, :, i]) for i in range(surface.shape[2])])
    start_frame, end_frame = np.min(np.where((is_surface_frames > 0) == True)), np.max(np.where((is_surface_frames > 0) == True))

    # step 2: clip predict
    for i in range(bin.shape[2]):
        if i <= start_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])
        elif i >= end_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    centers_source = ndimage.measurements.center_of_mass(surface)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    if len(np.unique(bin)) > 1:
        bin = threshold_by_otsu(bin, flatten=False)
        bin = getLargestCC(bin)
    bin = np.asarray(bin, np.float32)

    # step 3: clip outside radius
    centers = ndimage.measurements.center_of_mass(bin)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    # step 4: delete dots
    for k in range(bin.shape[2]):
        if (k < centers_source[2] - radius) or (k > centers_source[2] + radius):
            if len(np.unique(bin[:, :, k])) > 1:
                bin[:, :, k] = getLargestCC(bin[:, :, k])

    try:
        # step 5: delete dots2
        mask = np.zeros_like(bin[:, :, 0])
        mask[int(centers[0])-radius:int(centers[0])+radius, 
            int(centers[1])-radius:int(centers[1])+radius] = 1
        for k in range(bin.shape[2]):
            if (k < centers_source[2] - radius) or (k > centers_source[2] + radius):
                if len(np.unique(bin[:, :, k])) > 1:
                    bin[:, :, k] = mask*bin[:, :, k]
    except ValueError:
        print("ValueError: cannot convert float NaN to integer")

    return bin


def post_processing_results(bin, gt, surface, radius=6, pixel_spacing=np.array([64/180.,64/208., 64/300.])):
    is_surface_frames = np.array([np.count_nonzero(surface[:, :, i]) for i in range(surface.shape[2])])
    start_frame, end_frame = np.min(np.where((is_surface_frames > 0) == True)), np.max(np.where((is_surface_frames > 0) == True))

    # step 2: clip predict
    for i in range(bin.shape[2]):
        if i <= start_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])
        elif i >= end_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    centers_source = ndimage.measurements.center_of_mass(surface)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    if len(np.unique(bin)) > 1:
        bin = threshold_by_otsu(bin, flatten=False)
        bin = getLargestCC(bin)
    bin = np.asarray(bin, np.float32)

    # step 3: clip outside radius
    centers = ndimage.measurements.center_of_mass(bin)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    # step 4: delete dots
    for k in range(bin.shape[2]):
        if (k < centers_source[2] - radius) or (k > centers_source[2] + radius):
            if len(np.unique(bin[:, :, k])) > 1:
                bin[:, :, k] = getLargestCC(bin[:, :, k])

    try:
        # step 5: delete dots2
        mask = np.zeros_like(bin[:, :, 0])
        mask[int(centers[0])-radius:int(centers[0])+radius, 
            int(centers[1])-radius:int(centers[1])+radius] = 1
        for k in range(bin.shape[2]):
            if (k < centers_source[2] - radius) or (k > centers_source[2] + radius):
                if len(np.unique(bin[:, :, k])) > 1:
                    bin[:, :, k] = mask*bin[:, :, k]
    except ValueError:
        print("ValueError: cannot convert float NaN to integer")

    centers = ndimage.measurements.center_of_mass(bin)
    gt_centers = ndimage.measurements.center_of_mass(gt)
    dice_coeff = dice_coefficient_in_train(gt.flatten(), bin.flatten())
    le = np.linalg.norm((np.array(centers) - np.array(gt_centers)))*0.1

    # print(f"centers = {centers}, gt_centers = {gt_centers}, le = {le}")
    return bin, dice_coeff, le


def post_eval(bin, gt):
    centers = ndimage.measurements.center_of_mass(bin)
    gt_centers = ndimage.measurements.center_of_mass(gt)
    dice_coeff = dice_coefficient_in_train(gt.flatten(), bin.flatten())
    le = np.linalg.norm((np.array(centers) - np.array(gt_centers)))*0.1

    return dice_coeff, le

def post_processing_results2(bin, gt, surface, radius=5, pixel_spacing=np.array([64/180.,64/208., 64/300.])):
    is_surface_frames = np.array([np.count_nonzero(surface[:, :, i]) for i in range(surface.shape[2])])
    start_frame, end_frame = np.min(np.where((is_surface_frames > 0) == True)), np.max(np.where((is_surface_frames > 0) == True))

    # step 2: clip predict
    for i in range(bin.shape[2]):
        if i <= start_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])
        elif i >= end_frame:
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    centers_source = ndimage.measurements.center_of_mass(gt)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])


    bin = getLargestCC(threshold_by_otsu(bin, flatten=False))
    bin = np.asarray(bin, np.float32)

    # step 3: clip outside radius
    centers = ndimage.measurements.center_of_mass(bin)

    for i in range(bin.shape[2]):
        if (i < centers_source[2] - radius) or (i > centers_source[2] + radius):
            bin[:, :, i] = np.zeros_like(bin[:, :, i])

    dice_coeff = dice_coefficient_in_train(gt.flatten(), bin.flatten())
    gt_centers = ndimage.measurements.center_of_mass(gt)
    le = np.linalg.norm((np.array(centers) - np.array(gt_centers)) * np.array(pixel_spacing))

    return bin, dice_coeff, le


def errosion(x, iter=1):
    x = np.asarray(ndimage.binary_erosion(np.asarray(x, dtype=np.int), iterations=iter), dtype=np.int)
    return x