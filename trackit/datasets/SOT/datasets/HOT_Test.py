# trackit/datasets/SOT/datasets/HOT_Test.py

import os
import glob
import numpy as np

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class HOT_Test_Seed(BaseSeed):
    def __init__(self, root_path: str = None):
        if root_path is None:
            root_path = self.get_path_from_config('HOT_Test_PATH')
        super().__init__('HOT_Test', root_path)

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        seq_paths = sorted(glob.glob(os.path.join(self.root_path, '*')))
        constructor.set_total_number_of_sequences(len(seq_paths))
        constructor.set_bounding_box_format('XYWH')

        for seq_path in seq_paths:
            seq_name = os.path.basename(seq_path)
            gt_path  = os.path.join(seq_path, 'groundtruth_rect.txt')
            boxes    = np.loadtxt(gt_path, delimiter=None)

            with constructor.new_sequence() as seq:
                seq.set_name(seq_name)
                frames = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
                for idx, box in enumerate(boxes):
                    frame_path = frames[idx]
                    with seq.new_frame() as fr:
                        fr.set_path(frame_path)
                        fr.set_bounding_box(box, validity=True)
