# Risk1-B zero-fallback TODO

Risk1-B paper-table runs must not include deterministic fallback rows.

Current guardrail:

- Qwen first pass may be used.
- Qwen repair attempts may be used when they run in the same task context.
- If Qwen generation remains invalid after repair, the row is failed and must trigger prompt/parser tuning.
- `deterministic_locked_fields` fallback is diagnostic-only and invalidates the external-Qwen Risk1-B table row.
- A run with even one fallback row must not be reported as the paper-table Risk1-B Qwen result.

Reason:

Fallback rows turn the method into a Qwen plus handwritten-template hybrid, which does not answer the paper-table question: what is the LIBERO-10 success rate of Risk1-B external-Qwen alternative-goal candidate prompts?
