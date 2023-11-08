#######################################################################################################################
# Utility functions for visualisation and evaluation.                                                                 #
# Author:               Daniel Schirmacher                                                                            #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                  #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
#######################################################################################################################
from typing import Any, Dict, List

import numpy as np
import torch
from skimage.measure import label


def iou(pred: torch.Tensor, gt: torch.Tensor) -> List[np.ndarray]:
    """
    Compute Intersection over Union for groundtruth and prediction. Derived from:
    Caicedo_Carpenter (2019). Evaluation of deep learning strategies for nucleus segmentation in Fluorescence Images


    Parameters
    ----------

    pred: tensor of float32
        Tensor of shape (#samples, #channels, height, width) with elements in {0, 1}.

    gt: tensors of float32
        Tensor of shape (#samples, #channels, height, width) with elements in range [0, 1].


    Return
    ------

    Returns list of arrays, IoU, with len #samples.
    """
    assert (
        gt.size() == pred.size()
    ), f"Size of gt {gt.size()} and size of pred {pred.size()} are not matching."

    IOU = []

    for sample in zip(gt, pred):
        # convert semantic segmentation to instance segmentation
        gt_tmp, pred_tmp = sample[0].cpu().numpy(), sample[1].cpu().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1.0
        pred_tmp[pred_tmp < 0.5] = 0.0

        gt_tmp *= 255
        pred_tmp *= 255
        gt_tmp = gt_tmp[0, :, :].astype(np.uint8)
        pred_tmp = pred_tmp[0, :, :].astype(np.uint8)

        # convert semantic to instance segmentation
        gt_inst = label(gt_tmp)
        pred_inst = label(pred_tmp)

        # count objects and their area
        xedges, area_true = np.unique(gt_inst, return_counts=True)
        xedges = np.append(xedges, xedges[-1] + 1)
        yedges, area_pred = np.unique(pred_inst, return_counts=True)
        yedges = np.append(yedges, yedges[-1] + 1)

        # compute intersection
        h = np.histogram2d(
            gt_inst.flatten(), pred_inst.flatten(), bins=(xedges, yedges)
        )
        intersection = h[0]

        # compute union
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)
        union = area_true + area_pred - intersection

        # exclude background from the analysis
        intersection = intersection[1:, 1:]
        union = union[1:, 1:]

        # compute intersection over union
        union[union == 0] = 1e-9
        IOU.append(intersection / union)

    return IOU


def iou_to_f1(
    t_min: float, t_max: float, IOUs: List[np.ndarray]
) -> Dict[str, torch.Tensor]:
    """
    Compute f1 score for given IoU and threshold. Derived from:
    Caicedo_Carpenter (2019). Evaluation of deep learning strategies for nucleus segmentation in Fluorescence Images

    Parameter
    ---------

    t_min: float
        Float indicating IoU threshold at which segmentation errors objects are detected.

    t_max: float
        Float indicating IoU threshold at which predicted and ground truth segmentation overlap is a match.

    IOU: list of np.array of float
        List containing arrays of shape (#true_objects, #pred_objects) representing IoU for each sample in batch.


    Return
    ------

    Returns dict of torch.Tensor containing (float, int, int, int, int, int, int) representing f1-score, TP, FP, FN,
    splits, merges, and inaccurate masks.
    """
    f1 = []
    tp = []
    fp = []
    fn = []
    splits = []
    merges = []
    inaccurate_masks = []

    for i, IOU in enumerate(IOUs):
        # determine detected objects based on threshold
        matches_max = IOU > t_max
        matches_min = IOU > t_min
        matches_diff = np.logical_xor(matches_min, matches_max)

        # count TP, FN, FP
        true_positives = np.sum(matches_max, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches_min, axis=0) == 0  # Extra objects
        false_negatives = np.sum(matches_min, axis=1) == 0  # Missed objects

        assert np.all(np.less_equal(true_positives, 1))
        assert np.all(np.less_equal(false_positives, 1))
        assert np.all(np.less_equal(false_negatives, 1))

        tp.append(np.sum(true_positives))
        fp.append(np.sum(false_positives))
        fn.append(np.sum(false_negatives))

        # compute splits, merges
        gts, preds = np.where(matches_diff == 1)
        gt_ids, gt_counts = np.unique(gts, return_counts=True)
        pred_ids, pred_counts = np.unique(preds, return_counts=True)

        splits_tmp = sum(gt_counts > 1)
        merges_tmp = sum(pred_counts > 1)

        # compute inaccurate_masks
        # remove splits from gts and preds
        if splits_tmp > 0:
            for gt_id in gt_ids[gt_counts > 1]:
                bool_idx = gts != gt_id
                gts = gts[bool_idx]
                preds = preds[bool_idx]

        # remove merges from gts and preds
        if merges_tmp > 0:
            for pred_id in pred_ids[pred_counts > 1]:
                bool_idx = preds != pred_id
                gts = gts[bool_idx]
                preds = preds[bool_idx]

        inaccurate_masks_tmp = len(gts)

        splits.append(splits_tmp)
        merges.append(merges_tmp)
        inaccurate_masks.append(inaccurate_masks_tmp)

        # compute f1-score
        f1.append(
            2
            * tp[i]
            / (2 * tp[i] + fp[i] + fn[i] + inaccurate_masks[i] + 1e-9)
        )

    scores = {
        "f1": torch.FloatTensor(f1),
        "tp": torch.FloatTensor(tp),
        "fp": torch.FloatTensor(fp),
        "fn": torch.FloatTensor(fn),
        "splits": torch.FloatTensor(splits),
        "merges": torch.FloatTensor(merges),
        "inaccurate_masks": torch.FloatTensor(inaccurate_masks),
    }

    return scores


def remove_cells(
    original_image: np.array, labels_to_remove: List[int]
) -> torch.Tensor:
    """
    Remove cells with certain labels from a instant segmentation mask.

    Parameter
    ---------

    original_image: np.array
        Instant segmentation mask as np.array.

    labels_to_removes: List[int]
        List of labels that should be removed.


    Return
    ------

    Returns new instant segmentaion mask as torch.Tensor.
    """
    modified_cells = original_image.copy()

    is_part_of_labels_to_remove = np.isin(original_image, labels_to_remove)
    modified_cells[is_part_of_labels_to_remove] = 0

    modified_cells = torch.from_numpy(modified_cells)

    return modified_cells


def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> List[float]:
    """
    Compute IoU of ground truth cells with big and small cells equally weighted.

    Parameter
    ---------

    pred: tensor of float32
        Tensor of shape (#samples, #channels, height, width) with elements in {0, 1}.

    gt: tensors of float32
        Tensor of shape (#samples, #channels, height, width) with elements in range [0, 1].


    Return
    ------

    Returns torch.Tensor with IoU arrays with length #samples.
    """
    iou_list = []
    iou_small_list = []
    iou_big_list = []

    # loop through list of tensors
    for mask, mask_hat in zip(gt, pred):
        mask = mask.cpu().numpy()
        mask_hat = mask_hat.cpu()

        # generate images
        masks_inst = label(mask, background=0)
        cell_labels, sizes = np.unique(masks_inst, return_counts=True)
        cell_labels = cell_labels[1:]  # remove background
        sizes = sizes[1:]

        # get big cell labels
        is_big = np.where(sizes > 2000)[0]
        big_labels = cell_labels[is_big]

        # get small cell labels
        is_small = np.where(sizes <= 2000)[0]
        small_labels = cell_labels[is_small]

        # calculate iou
        small_cells = remove_cells(masks_inst, big_labels)
        big_cells = remove_cells(masks_inst, small_labels)
        small_big_tensor = torch.stack((small_cells, big_cells), 0)
        mask_hat_tensor = torch.stack((mask_hat, mask_hat), 0)
        ious = iou(
            mask_hat_tensor, small_big_tensor
        )  # columns=predictions, rows=ground truths

        small_iou = ious[0]
        big_iou = ious[1]

        # take ious according to gt
        small_iou = np.amax(small_iou, axis=1, initial=0)
        big_iou = np.amax(big_iou, axis=1, initial=0)

        if len(big_labels) == 0 and len(small_labels) > 0:
            avg_IOU_small_cells = np.mean(small_iou)
            avg_IOU_big_cells = 0
            avg_iou = avg_IOU_small_cells
        elif len(small_labels) == 0 and len(big_labels) > 0:
            avg_IOU_big_cells = np.mean(big_iou)
            avg_IOU_small_cells = 0
            avg_iou = avg_IOU_big_cells
        elif len(small_labels) == 0 and len(big_labels) == 0:
            avg_IOU_big_cells = 0
            avg_IOU_small_cells = 0
            avg_iou = 0
        else:
            avg_IOU_small_cells = np.mean(small_iou)
            avg_IOU_big_cells = np.mean(big_iou)
            avg_iou = (avg_IOU_big_cells + avg_IOU_small_cells) / 2

        iou_list.append(avg_iou)
        iou_big_list.append(avg_IOU_big_cells)
        iou_small_list.append(avg_IOU_small_cells)

    data = {
        "iou": torch.FloatTensor(iou_list),
        "iou_big": torch.FloatTensor(iou_big_list),
        "iou_small": torch.FloatTensor(iou_small_list),
    }

    return data


def compute_f1(
    pred: torch.Tensor, gt: torch.Tensor, t_min: float, t_max: float
) -> Dict[str, List[Any]]:
    """
    Wrapper around iou and iou_to_f1.


    Parameters
    ----------

    pred: tensor of float32
        Tensor of shape (#samples, #channels, height, width) with elements in {0, 1}.

    gt: tensors of float32
        Tensor of shape (#samples, #channels, height, width) with elements in range [0, 1].

    t_min: float
        Float indicating IoU threshold at which segmentation errors objects are detected.

    t_max: float
        Float indicating IoU threshold at which predicted and ground truth segmentation overlap is a match.


    Return
    ------

    Returns dict of torch.Tensor containing (float, int, int, int, int, int, int) representing f1-score, TP, FP, FN,
    splits, merges, and inaccurate masks.
    """
    IOU = iou(pred, gt)
    return iou_to_f1(t_min, t_max, IOU)
