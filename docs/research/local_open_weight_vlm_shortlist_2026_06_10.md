# Local Open-Weight VLM Shortlist for Subgoal Generation

Date: 2026-06-10

## Goal

Find open-weight vision-language models that are:

- strong enough for image-grounded subgoal generation,
- light enough to be realistic on a local Mac,
- and practical to run through `llama.cpp` or standard local inference stacks.

## Best Practical Shortlist

### 1. Qwen2.5-VL-7B-Instruct

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Why it stands out:
  - strong general vision-language quality for a still-manageable size,
  - explicit support for visual localization with boxes or points,
  - strong document/chart/text-in-image performance,
  - Apache 2.0 license.
- Why it fits subgoal generation:
  - easiest mainstream choice if we want structured image-grounded outputs like:
    - `next_subgoal`,
    - `target_object`,
    - `target_region`,
    - `point` / `bbox`,
    - `success_check`.
- Local path:
  - Hugging Face model page exposes quantizations for `llama.cpp`.

### 2. Qwen2.5-VL-3B-Instruct

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Why it stands out:
  - same family and interface as the 7B model,
  - much lighter for local iteration,
  - still supports localization and structured outputs.
- Why it fits subgoal generation:
  - best first probe when we care about fast local iteration over maximum accuracy.
- Local path:
  - Hugging Face model page exposes quantizations for `llama.cpp`.

### 3. Gemma 3 4B IT

- Model: `google/gemma-3-4b-it`
- Why it stands out:
  - lightweight multimodal model family,
  - broad multilingual and long-context support,
  - strong open small-model baseline,
  - quantizations available for `llama.cpp`.
- Why it fits subgoal generation:
  - good small general-purpose VLM if we want compact inference and decent reasoning.
- Caveat:
  - license is `gemma`, not Apache 2.0.
  - less directly “robot-grounding flavored” than Qwen2.5-VL.

### 4. InternVL3-2B

- Model: `OpenGVLab/InternVL3-2B`
- Why it stands out:
  - very small recent multimodal model,
  - Apache-2.0/MIT-style open distribution path,
  - explicit emphasis on tool use, GUI grounding, spatial reasoning, and multimodal perception.
- Why it fits subgoal generation:
  - attractive if we want a compact modern VLM with stronger “agentic” or grounding-oriented flavor than many older 2B-class models.
- Caveat:
  - less battle-tested in local hobby stacks than Qwen/MiniCPM.

### 5. MiniCPM-V 2.6

- Model: `openbmb/MiniCPM-V-2_6`
- Why it stands out:
  - very strong efficiency story,
  - official local-device focus,
  - official `llama.cpp` and GGUF support path,
  - strong OCR and multi-image/video support,
  - competitive single-image quality for 8B.
- Why it fits subgoal generation:
  - especially good if local deployment convenience matters as much as benchmark quality.
- Caveat:
  - 8B is still heavier than the 2B-4B class.

### 6. MolmoE-1B / Molmo 7B-D

- Models:
  - `allenai/MolmoE-1B-0924`
  - `allenai/Molmo-7B-D-0924`
- Why they stand out:
  - fully open Apache 2.0,
  - good research credibility,
  - good human-eval reputation,
  - quantization path exists for `llama.cpp`.
- Why they fit subgoal generation:
  - `MolmoE-1B` is the tiny “worth trying” option.
  - `Molmo-7B-D` is the stronger AllenAI option if we can afford a bigger model.
- Caveat:
  - local ecosystem convenience is weaker than Qwen/MiniCPM.

## Not First Choice

### PaliGemma 2 3B

- Model: `google/paligemma2-3b-pt-224`
- Why not first:
  - the model card explicitly says this checkpoint is a base model and is recommended after downstream fine-tuning.
- Good use:
  - only if we are willing to fine-tune for a tightly scoped subgoal format.

## Recommended Order

If the goal is "best practical local open-weight VLM for subgoal generation" on this Mac:

1. `Qwen/Qwen2.5-VL-7B-Instruct`
2. `Qwen/Qwen2.5-VL-3B-Instruct`
3. `google/gemma-3-4b-it`
4. `OpenGVLab/InternVL3-2B`
5. `openbmb/MiniCPM-V-2_6`
6. `allenai/MolmoE-1B-0924`

## Suggested First Evaluation

For this repo, the cleanest first pass is:

1. Try `Qwen2.5-VL-3B-Instruct` as the lightweight baseline.
2. Try `Qwen2.5-VL-7B-Instruct` as the stronger reference.
3. Try `Gemma 3 4B IT` or `InternVL3-2B` as a compact alternative.

Use the same prompt schema for all models:

- `task_instruction`
- `current_observation`
- `current_failure_or_progress`
- `required_output_json`

And ask for:

- `subgoal_text`
- `target_object`
- `target_region_or_point`
- `stop_condition`
- `confidence`

## Source Notes

- Qwen2.5-VL pages show:
  - localization by bounding boxes or points,
  - structured outputs,
  - Apache 2.0,
  - quantization path for `llama.cpp`.
- Gemma 3 4B page shows:
  - lightweight multimodal design,
  - 4B parameters,
  - quantization path for `llama.cpp`.
- InternVL3-2B page shows:
  - 2B family member,
  - strong grounding/tool-use positioning,
  - quantization path for `llama.cpp`.
- MiniCPM-V 2.6 page shows:
  - 8B total parameters,
  - explicit `llama.cpp` and GGUF support,
  - strong local-efficiency claims.
- Molmo pages show:
  - open Apache 2.0 licensing,
  - quantization path for `llama.cpp`,
  - viable 1B and 7B options.
