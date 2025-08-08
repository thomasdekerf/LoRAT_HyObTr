import argparse
import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

from trackit.data.components.result_collector.handler.one_pass_evaluation.ope_metrics import (
    DatasetOPEMetricsListBuilder,
    compute_one_pass_evaluation_metrics,
)
from trackit.miscellanies.parser.txt import load_numpy_array_from_txt
from trackit.core.operator.numpy.bbox.format import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_xywh,
)
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import (
    get_siamfc_cropping_params,
    apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
)
from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
    SiamFCCroppingParameterSimpleProvider,
)
from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
    PostProcessing_BoxWithScoreMap,
)
from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import (
    get_foreground_bounding_box,
)
from trackit.models.methods.LoRAT.builder import create_LoRAT_build_context
from trackit.models import ModelManager


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


def _build_model(weight_path: str, device: torch.device) -> torch.nn.Module:
    config = {
        "type": "LoRAT",
        "common": {
            "template_feat_size": [8, 8],
            "search_region_feat_size": [16, 16],
        },
        "model": {
            "type": "dinov2",
            "backbone": {"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
            "lora": {"r": 64, "alpha": 64, "dropout": 0.0, "use_rslora": False},
        },
    }
    build_ctx = create_LoRAT_build_context(config)
    manager = ModelManager(build_ctx)
    manager.load_state_dict_from_file(weight_path, strict=False, print_missing=False)
    model = manager.create(device).model
    model.eval()
    return model


def evaluate_split(data_root: str, split: str, weight_path: str, device: str) -> None:
    device_ = torch.device(device)
    model = _build_model(weight_path, device_)

    template_size = np.array([112, 112])
    search_size = np.array([224, 224])
    template_feat_size = np.array([8, 8])
    stride = template_size / template_feat_size

    norm_fn = get_dataset_norm_stats_transform("imagenet", inplace=True)
    post_process = PostProcessing_BoxWithScoreMap(device_, (16, 16), (224, 224), 0.45)
    post_process.start()

    split_dir = os.path.join(data_root, split)
    sequences = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    sequences.sort()

    metrics_builder = DatasetOPEMetricsListBuilder()

    for seq in sequences:
        seq_dir = os.path.join(split_dir, seq)
        image_paths, gt_cxcywh = _read_sequence(seq_dir)
        gt_xyxy = bbox_cxcywh_to_xyxy(gt_cxcywh)

        preds = []
        times: List[float] = []

        # initialization
        first_image = np.array(Image.open(image_paths[0])).astype(np.float32)
        init_bbox = gt_xyxy[0]
        init_tensor = torch.from_numpy(first_image).permute(2, 0, 1).to(device_)
        template_crops = get_siamfc_cropping_params(init_bbox, 2.0, template_size)
        z, z_mean, template_crops = apply_siamfc_cropping(
            init_tensor, template_size, template_crops, "bilinear", False
        )
        z = z / 255.0
        norm_fn(z)
        template = z.unsqueeze(0)
        template_mean = z_mean

        fg_bbox = get_foreground_bounding_box(init_bbox, template_crops, stride)
        mask = torch.zeros(template_feat_size[1], template_feat_size[0], dtype=torch.long)
        mask[fg_bbox[1]: fg_bbox[3], fg_bbox[0]: fg_bbox[2]] = 1
        z_feat_mask = mask.unsqueeze(0).to(device_)

        cropping_provider = SiamFCCroppingParameterSimpleProvider(4.0, 10)
        cropping_provider.initialize(init_bbox)

        preds.append(init_bbox)
        times.append(0.0)

        for img_path in image_paths[1:]:
            image = np.array(Image.open(img_path)).astype(np.float32)
            tensor = torch.from_numpy(image).permute(2, 0, 1).to(device_)
            start = time.time()
            crop_params = cropping_provider.get(search_size)
            x, _, crop_params = apply_siamfc_cropping(
                tensor, search_size, crop_params, "bilinear", False, template_mean
            )
            x = x / 255.0
            norm_fn(x)
            x = x.unsqueeze(0)
            with torch.inference_mode():
                out = model(z=template, x=x, z_feat_mask=z_feat_mask)
            processed = post_process(out)
            bbox_crop = processed["box"][0].cpu().numpy()
            conf = float(processed["confidence"][0].cpu())
            bbox_img = apply_siamfc_cropping_to_boxes(bbox_crop, crop_params)
            cropping_provider.update(conf, bbox_img, np.array([image.shape[1], image.shape[0]]))
            times.append(time.time() - start)
            preds.append(bbox_img)

        preds = np.array(preds, dtype=np.float64)
        metrics, _ = compute_one_pass_evaluation_metrics(
            preds,
            gt_xyxy,
            None,
            np.array(times, dtype=np.float64),
        )
        metrics_builder.append(seq, metrics)

    post_process.stop()

    dataset_metrics = metrics_builder.build().get_mean()
    precision_at_20 = dataset_metrics.precision_score
    success_auc = dataset_metrics.success_score
    print(f"Precision@20px: {precision_at_20:.4f}")
    print(f"Success AUC: {success_auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HOT dataset inference and evaluation")
    parser.add_argument("data_root", type=str, help="Path to HOT dataset root")
    parser.add_argument("weights", type=str, help="Path to pretrained .bin weight file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()
    evaluate_split(args.data_root, args.split, args.weights, args.device)


if __name__ == "__main__":
    main()
