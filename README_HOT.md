# HOT Dataset Support

This repository has been extended with a minimal interface for the **HOT**
(Hyper-Object Tracking) style dataset. The dataset is expected to be laid out
as follows:

```
HOT_ROOT/
  train/
    <sequence-name>/
      0001.jpg
      0002.jpg
      ...
      groundtruth_rect.txt
  test/
    <sequence-name>/
      0001.jpg
      0002.jpg
      ...
      groundtruth_rect.txt   # only the first line is used to initialise
```

Each `groundtruth_rect.txt` contains four values per line representing the
bounding box centre `(cx, cy)` and its `(w, h)`. The file may use a mixture of
spaces and tab characters as delimiters – this is handled automatically by the
loader.

## Inference and Evaluation

1. Download the desired LoRAT `.bin` weight file and place it in a folder such
   as `weights/` inside this repository.
2. Run the evaluation script, providing the dataset root and weight path:

```bash
python hot_inference.py /path/to/HOT_ROOT weights/lorat_model.bin --split test --device cuda:0
```

The script loads the tracker, runs it on every sequence and prints the
precision at 20 pixels and the success AUC over all sequences.

## Fine‑tuning on HOT

To fine‑tune the tracker starting from an existing checkpoint, use
`hot_train.py`. The script is a light wrapper around `main.py` and accepts all
other arguments supported by the original entry point. Pretrained weights can
be `.bin` files downloaded from the project page.

```bash
python hot_train.py <method_name> <config_name> /path/to/HOT_ROOT \
       /path/to/pretrained_weights.pth /path/to/output_dir
```

- `method_name` and `config_name` correspond to a configuration file under
  `config/<method_name>/<config_name>.yaml`.
- The dataset root is exported via the `HOT_PATH` environment variable; adjust
  the path if your data lives elsewhere.
- Checkpoints and logs are written to the specified output directory.

## Notes

Ensure the `HOT_PATH` environment variable always points to your dataset root
when training or evaluating on HOT.
