# X-VLA MPS Mac Port

This folder is a standalone Mac-local X-VLA MPS smoke/e2e inference port.

It uses the public LeRobot X-VLA implementation and checkpoint:

- Code path: `lerobot.policies.xvla`
- Checkpoint: `lerobot/xvla-base`
- Working Mac MPS runtime found on this machine: `torch==2.4.1`, `torchvision==0.19.1`

The repo's newer `torch` environments reported `mps_available=False` on macOS 26.5.
This separate venv keeps the working MPS combo isolated.

## Setup

```sh
sh experiments/xvla_mps_mac_port/scripts/setup_env.sh
```

The default venv path is:

```text
.venv-xvla-mps-py312
```

## E2E Inference

Run with normal macOS Metal access:

```sh
.venv-xvla-mps-py312/bin/python \
  experiments/xvla_mps_mac_port/tools/run_xvla_mps_inference.py \
  --local-files-only \
  --output experiments/xvla_mps_mac_port/runs/xvla_mps_e2e.json
```

## Port Notes

LeRobot `lerobot/xvla-base` loads on MPS. The important compatibility detail is
the language tokenizer length.

The public 2toINF X-VLA processor uses:

```python
language_max_length = 50
```

and tokenizes with:

```python
padding="max_length", max_length=self.language_max_length, truncation=True
```

LeRobot's checkpoint config currently reports `tokenizer_max_length=1024`.
If the LeRobot preprocessor is used with that value, the action transformer sees
a sequence longer than its 512-position embedding:

```text
Sequence length ... exceeds max_len_seq=512
```

This port follows the public X-VLA processor behavior by setting
`cfg.tokenizer_max_length = 50`. It does not truncate the visual token stream or
replace `forward_vlm()`.

The script keeps the public inference shape fixed:

```text
num_views = 3
image_size = 256
language_max_length = 50
steps = 10
```

The MPS-specific changes are limited to:

```text
torch==2.4.1
torchvision==0.19.1
cfg.device = "mps"
cfg.dtype = "float32"
```

No model forward method, action transformer, Florence encoder, visual token
stream, or action post-processing is monkey-patched.
