import os
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.miscellanies.parser.txt import load_numpy_array_from_txt


def construct_HOT(constructor: SingleObjectTrackingDatasetConstructor, seed) -> None:
    """Construct the HOT dataset.

    The expected folder structure is::

        root/train/<sequence>/
        root/test/<sequence>/

    Each sequence folder must contain the frames and a ``groundtruth_rect.txt``
    file with annotations encoded as ``cx cy w h`` using whitespace
    separators (tabs or spaces). The test split may omit this file.
    """
    constructor.set_bounding_box_format('XYWH')

    root_path = seed.root_path
    split_path = os.path.join(root_path, seed.split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f'Cannot find split {seed.split} under {root_path}')

    sequence_names = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    sequence_names.sort()

    for seq_name in sequence_names:
        seq_dir = os.path.join(split_path, seq_name)
        image_files = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        gt_path = os.path.join(seq_dir, 'groundtruth_rect.txt')
        if os.path.isfile(gt_path):
            bbox_anno = load_numpy_array_from_txt(gt_path, delimiter=None)
            if bbox_anno.ndim == 1:
                bbox_anno = bbox_anno[None, :]
        else:
            bbox_anno = np.zeros((len(image_files), 4), dtype=np.float64)
        if len(bbox_anno) != len(image_files):
            raise ValueError(f'Number of images and annotations do not match for {seq_name}')

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(seq_name)
            seq_rel_path = os.path.join(seed.split, seq_name)
            for img_name, bbox in zip(image_files, bbox_anno):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(seq_rel_path, img_name))
                    cx, cy, w, h = bbox
                    x = cx - w / 2.0
                    y = cy - h / 2.0
                    frame_constructor.set_bounding_box([x, y, w, h])
