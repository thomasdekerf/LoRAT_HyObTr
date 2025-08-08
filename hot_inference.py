import argparse
import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

from trackit.data.components.result_collector.handler.one_pass_evaluation.ope_metrics import (
    DatasetOPEMetricsListBuilder,
    compute_one_pass_evaluation_metrics,
)
from trackit.miscellanies.parser.txt import load_numpy_array_from_txt


def _read_sequence(sequence_path: str) -> Tuple[List[str], np.ndarray]:
    images = sorted(
        [
            os.path.join(sequence_path, f)
            for f in os.listdir(sequence_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    gt_path = os.path.join(sequence_path, "groundtruth_rect.txt")
    ann = load_numpy_array_from_txt(gt_path, delimiter=None)
    if ann.ndim == 1:
        ann = ann[None, :]
    return images, ann


class DummyTracker:
    """A placeholder tracker.

    Replace this class with the real tracker. It simply returns the
    ground-truth box for every frame, serving as an example of how to
    integrate a tracker with the evaluation utilities."""

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        self._bbox = bbox

    def track(self, image: np.ndarray) -> np.ndarray:
        return self._bbox


def evaluate_split(data_root: str, split: str) -> None:
    split_dir = os.path.join(data_root, split)
    sequences = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    sequences.sort()

    tracker = DummyTracker()
    metrics_builder = DatasetOPEMetricsListBuilder()

    for seq in sequences:
        seq_dir = os.path.join(split_dir, seq)
        image_paths, gt = _read_sequence(seq_dir)
        preds = []
        times: List[float] = []
        for i, (img_path, bbox) in enumerate(zip(image_paths, gt)):
            image = np.array(Image.open(img_path))
            start = time.time()
            if i == 0:
                tracker.initialize(image, bbox)
                pred = bbox
            else:
                pred = tracker.track(image)
            times.append(time.time() - start)
            preds.append(pred)
        preds = np.array(preds, dtype=np.float64)
        gt = np.array(gt, dtype=np.float64)
        metrics, _ = compute_one_pass_evaluation_metrics(
            preds,
            gt,
            None,
            np.array(times, dtype=np.float64),
        )
        metrics_builder.append(seq, metrics)

    dataset_metrics = metrics_builder.build().get_mean()
    precision_at_20 = dataset_metrics.precision_score
    success_auc = dataset_metrics.success_score
    print(f"Precision@20px: {precision_at_20:.4f}")
    print(f"Success AUC: {success_auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HOT dataset inference and evaluation")
    parser.add_argument("data_root", type=str, help="Path to HOT dataset root")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()
    evaluate_split(args.data_root, args.split)


if __name__ == "__main__":
    main()
