import numpy as np
import torch

from cellseg.utils.evaluation import iou, iou_to_f1


def test_iou():
    """
    test if iou returns as expected.
    """
    pred = torch.FloatTensor(
        [
            [
                [
                    [1.0, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.9],
                    [0.0, 0.0, 0.0, 0.3],
                    [0.9, 0.4, 0.0, 0.8],
                ]
            ]
        ]
    )

    gt = torch.FloatTensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0],
                ]
            ]
        ]
    )

    IOU = iou(pred, gt)
    IOU_expected = [
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.25], [0.0, 0.0, 0.5, 0.0]])
    ]

    assert (IOU[0] == IOU_expected[0]).all()


def test_iou_to_f1():
    """
    tests iou to f1 conversion
    """
    IOU = [
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.25, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            ]
        )
    ]

    results = iou_to_f1(t_min=0.1, t_max=0.6, IOUs=IOU)
    results_expected = {
        "f1": torch.FloatTensor([2 / (5 + 1e-9)]),
        "tp": torch.FloatTensor([1]),
        "fp": torch.FloatTensor([1]),
        "fn": torch.FloatTensor([1]),
        "splits": torch.FloatTensor([1]),
        "merges": torch.FloatTensor([1]),
        "inaccurate_masks": torch.FloatTensor([1]),
    }

    assert torch.isclose(results["f1"], results_expected["f1"])
    assert torch.equal(results["tp"], results_expected["tp"])
    assert torch.equal(results["fp"], results_expected["fp"])
    assert torch.equal(results["fn"], results_expected["fn"])
    assert torch.equal(results["splits"], results_expected["merges"])
    assert torch.equal(results["merges"], results_expected["merges"])
    assert torch.equal(
        results["inaccurate_masks"], results_expected["inaccurate_masks"]
    )
