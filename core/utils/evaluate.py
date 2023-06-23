import numpy as np

from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure, exposure
from tqdm import tqdm

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def threshold_by_otsu(pred_vessels, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_vessels)
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def misc_measures_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    try:
        acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    except:
        acc = 0.

    try:
        sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    except:
        sensitivity = 0.

    try:
        specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    except:
        specificity = 0.
    return acc, sensitivity, specificity


def average_surface_distance(predictions, labels, spacing=[1, 1, 1]):
    """
    calculate average surface distance
    :param predictions:
    :param labels:
    :param spacing:
    :return:
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    pred_img = sitk.GetImageFromArray(np.asarray(predictions, np.float64))
    pred_img.SetSpacing(spacing)
    lab_img = sitk.GetImageFromArray(np.asarray(labels, np.float64))
    lab_img.SetSpacing(spacing)
    hausdorff_distance_filter.Execute(pred_img, lab_img)
    asd = hausdorff_distance_filter.GetAverageHausdorffDistance()
    return asd



def hausdorff_distance(predictions, labels, spacing=[1, 1, 1]):
    """
    calculate hausdorff distance from prediction verse labels
    :param predictions: 3D Array
    :param labels: 3D Array
    :param spacing: voxel spacing
    :return:
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    pred_img = sitk.GetImageFromArray(predictions)
    pred_img.SetSpacing(spacing)
    lab_img = sitk.GetImageFromArray(np.asarray(labels, dtype=np.float64))
    lab_img.SetSpacing(spacing)
    hausdorff_distance_filter.Execute(pred_img, lab_img)
    hd = hausdorff_distance_filter.GetHausdorffDistance()
    return hd


def evaluate_single_image_distance(pred_image, true_image, spacing=[1,1,1]):
    hd = hausdorff_distance(pred_image, true_image, spacing)
    asd = average_surface_distance(pred_image, true_image, spacing)
    return hd, asd


# def evaluate_source_recon(pred_image, true_image):
#     pred_image_vec = pred_image.flatten()
#     true_image_vec = true_image.flatten()
#     mse = np.linalg.norm(pred_image_vec - true_image_vec)
#     pass



def evaluate_single_image(pred_image, true_image):
    #assert len(pred_image.shape) == 4
    #assert len(true_image.shape) == 4

    pred_image_vec = pred_image.flatten()
    true_image_vec = true_image.flatten()

    try:
        binary_image = threshold_by_otsu(pred_image, flatten=False)
    except:
        binary_image = np.zeros_like(true_image)
    binary_image_vec = binary_image.flatten()

    dice_coeff = dice_coefficient_in_train(true_image_vec, binary_image_vec)
    # acc, sensitivity, specificity = misc_measures_in_train(true_image_vec, binary_image_vec)
    #
    # mse = np.linalg.norm(pred_image_vec - true_image_vec)
    #
    # binary_image_gt = np.zeros_like(true_image)
    # binary_image_gt[true_image>0] = 1
    #
    # hd, asd = evaluate_single_image_distance(binary_image, binary_image_gt, spacing=[1,1,1])

    #return binary_image, dice_coeff, acc, sensitivity, specificity, mse, hd, asd
    return binary_image, dice_coeff