# Training Pipeline Skeleton

This directory contains dry-run friendly training entry points for
`training-001`. It does not download datasets, does not download pretrained
weights, and does not claim mAP, latency, ADE, FDE, or model size unless those
values are measured from a local run.

## Detection

YOLO-format data is described in `detection/data.yaml`:

```text
datasets/haiyu_yolo_placeholder/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Each label row is `class_id x_center y_center width height` normalized to
`[0,1]`. The manifest lists 8 required scenarios: clear day, night low light,
fog/haze, rain, backlight/glare, port clutter, occlusion, and long-range small
targets. Public or synthetic sources remain placeholders until reviewed.

Commands:

```bash
python training/detection/train_yolo.py --dry-run
python training/detection/prune_yolo.py --dry-run
python training/detection/export_onnx_rknn.py --dry-run
```

Non-dry runs require local dataset files and local weights. The scripts reject
missing local weights instead of triggering automatic downloads. Ultralytics
YOLOv8 is AGPL-3.0; distribution or network use must review license duties.

## Trajectory

The trajectory training script builds the required LSTM + causal attention
model and uses PINN loss components during training. Input feature rows follow
the algorithm contract:

```text
[x, y, vx, vy, ax, ay, dt]
```

Dry-run commands:

```bash
python training/trajectory/train_lstm_pinn.py --dry-run
python training/trajectory/evaluate_prediction.py --dry-run
```

For real local training, provide a JSON list of trajectories or a dict with a
`trajectories` key. Split by trajectory before windowing to avoid overlapping
window leakage between train and validation sets.

## Metric Table Format

Do not fill numeric values until measured locally.

| Model | Dataset split | mAP@0.5 | Latency ms | Size MB | Source layer |
|---|---|---:|---:|---:|---|
| YOLOv8n baseline | val | 拟开展/未验证 | 拟开展/未验证 | 拟开展/未验证 | local |
| Pruned INT8 RKNN | test | 拟开展/未验证 | 拟开展/未验证 | 拟开展/未验证 | local |

For trajectory prediction:

| Model | Split | ADE | FDE | Source layer |
|---|---|---:|---:|---|
| LSTM causal attention + PINN | val | 拟开展/未验证 | 拟开展/未验证 | local |

## Metric Boundaries

Platform metrics from the project contract and prototype metrics are different
layers. Prototype numeric values from the contract must be labeled
`样机级，未经实港验证` whenever repeated. Unmeasured training outputs in this
directory must stay as `拟开展/未验证`.
