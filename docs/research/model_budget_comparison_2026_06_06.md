# Model Budget Comparison for SmolVLA Affordance Layer

Date: 2026-06-06

Reference runtime actor:

- SmolVLA: 450M parameters.
- Reported in Hugging Face materials as a compact VLA for real-time robotics.
- Related model-card evidence reports roughly 858-908 MB FP16/inference memory
  and about 169 ms latency on H200-class measurement for SmolVLA-450M; the exact
  latency will differ on Mac, RTX 3090/4070, and RunPod GPUs.

Design rule:

The runtime affordance/perception module should be smaller than SmolVLA and
preferably much smaller:

- Hard ceiling: <= 450M parameters.
- Practical target: <= 100M parameters.
- Preferred target: 5M-30M parameters.
- Runtime memory target: <= 500 MB, preferably <= 150 MB.
- Latency target: less than one SmolVLA action-chunk inference, preferably
  5-30 ms on GPU or accelerator-class hardware.

## Candidate Budget Table

| Candidate | Role | Params | Approx size / memory evidence | Latency evidence | SmolVLA ratio | Runtime verdict |
| --- | --- | ---: | --- | --- | ---: | --- |
| SmolVLA | Base actor | 450M | around 858-908 MB FP16/inference evidence | around 169 ms on H200 evidence; other public pages claim real-time robotics | 1.00x | Reference |
| YOLO-Worldv2-n | Open-vocab detector | 3.16M | small detector weights | real-time YOLO family; source reports YOLO-World 52 FPS on V100 for LVIS setting | 0.007x | Strong runtime candidate |
| YOLO-Worldv2-s | Open-vocab detector | 11.17M | small detector weights | real-time family | 0.025x | Strong runtime candidate |
| YOLO-Worldv2-m | Open-vocab detector | 25.90M | moderate detector weights | real-time family | 0.058x | Possible if accuracy needed |
| YOLO-Worldv2-l | Open-vocab detector | 43.69M | moderate detector weights | real-time family | 0.097x | Still under budget, but less ideal |
| YOLO-Worldv2-x | Open-vocab detector | 68.23M | larger detector weights | real-time family | 0.152x | Under hard budget, not first choice |
| YOLOE-v8/11-s | Open-vocab detect+segment | about 26M by Ultralytics docs | around small YOLO segmentation class | docs claim matching YOLO11 speed/params after reparameterization | 0.058x | Strong if segmentation is needed |
| MobileSAM | Prompted segmenter | about 10-13M total depending accounting | Qualcomm: encoder 26.6 MB, decoder 23.7 MB; memory 0-192 MB on Snapdragon profile | Qualcomm: 2.15 ms encoder on Snapdragon NPU profile | 0.02-0.03x | Strong segmentation fallback |
| EfficientSAM 10M | Prompted segmenter | 10M | osam reports 40 MB | efficient SAM variants report large speedups over SAM | 0.022x | Strong segmentation fallback |
| EfficientSAM 30M | Prompted segmenter | 26M | osam reports 100 MB | efficient SAM variants report large speedups over SAM | 0.058x | Strong segmentation fallback |
| SAM2 Tiny | Prompted segmenter/video | 39M | osam reports 150 MB | real-time-oriented but heavier than MobileSAM/EfficientSAM | 0.087x | Possible, not first choice |
| SAM2 Small | Prompted segmenter/video | 46M | osam reports 170 MB | heavier than tiny | 0.102x | Possible, not first choice |
| SAM2 Base+ | Prompted segmenter/video | 82M | osam reports 300 MB | heavier | 0.182x | Borderline for our philosophy |
| SAM2 Large | Prompted segmenter/video | 227M | osam reports 870 MB | heavy | 0.504x | Avoid runtime |
| SAM ViT-B | Prompted segmenter | 91-94M | osam reports about 100 MB for SAM 100M | source examples report 0.3 s GPU / 1.5 s CPU in microscopy setting | 0.20x | Offline/reference, not preferred runtime |
| SAM ViT-L | Prompted segmenter | 308-313M | osam reports about 310 MB | slower | 0.68x | Offline only |
| SAM ViT-H | Prompted segmenter | 632-642M | osam reports about 630 MB; original largest SAM | heavy; full SAM is known runtime bottleneck | 1.4x | Reject runtime |
| Grounding DINO Tiny | Open-vocab detector | about 173M total in one ICLR table | larger than YOLO-World/YOLOE | grounding models are generally slower than YOLO-style detectors | 0.38x | Offline pseudo-labeler, not runtime first |
| Grounding DINO Base | Open-vocab detector | about 233M total in one ICLR table | large | slower than YOLO-style detectors | 0.52x | Offline pseudo-labeler |
| MobileNetV3-Small affordance head | Trainable custom predictor | 2.5-3M backbone plus small head | tiny | mobile latency evidence around tens of ms on phone-class devices | 0.006x | Best final runtime direction |
| EfficientNet-Lite0 affordance head | Trainable custom predictor | 4.7M backbone plus head | tiny | TF reports mobile latencies around 6-12 ms depending device/profile | 0.010x | Best final runtime direction |
| ResNet18 affordance head | Trainable custom predictor | 11.7M plus head | about 45 MB FP32 weights | CPU/GPU friendly | 0.026x | Strong simple baseline |
| DINOv2 ViT-S affordance head | Feature extractor + head | 21M plus head | moderate | heavier than MobileNet/ResNet but strong features | 0.047x | Strong if visual generality matters |

## Practical Ranking

Runtime main candidates:

1. Oracle projection: 0M, no model, best first experiment.
2. MobileNetV3-Small or EfficientNet-Lite0 affordance heatmap head: 3M-5M.
3. ResNet18 affordance head: 12M, simple and defensible.
4. DINOv2 ViT-S affordance head: 21M, stronger visual features but heavier.
5. YOLO-Worldv2-n/s detector-only affordance: 3M-11M.
6. YOLOE-s if segmentation is needed: about 26M.
7. MobileSAM / EfficientSAM only when mask quality matters.

Offline or reference-only:

1. Grounding DINO Tiny/Base.
2. SAM ViT-B/L/H.
3. SAM2 Base+/Large.

Rejected as runtime defaults:

- Grounding DINO + full SAM/SAM2 large stack.
- Any perception stack whose combined params or memory approach/exceed SmolVLA.

## Recommended Experiment Stack

Experiment 0:

- SmolVLA raw image.

Experiment 1:

- SmolVLA + oracle projected point/crop.
- Added params: 0M.

Experiment 2:

- SmolVLA + tiny learned affordance head.
- Candidate backbones: MobileNetV3-Small, EfficientNet-Lite0, ResNet18.
- Added params: 3M-12M.

Experiment 3:

- SmolVLA + YOLO-Worldv2-s bbox geometry.
- Added params: about 11M.

Experiment 4:

- SmolVLA + YOLOE-s or EfficientSAM only if segmentation is necessary.
- Added params: about 10M-26M for lightweight segmenter; avoid full SAM.

## Design Conclusion

The paper should explicitly state that the proposed affordance layer is not a
second large foundation model. The runtime layer should remain one to two orders
of magnitude smaller than SmolVLA, and heavy models should only be used for
offline pseudo-labeling or oracle/reference studies.

