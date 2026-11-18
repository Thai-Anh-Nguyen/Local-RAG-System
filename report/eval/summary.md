# Evaluation summary (automatic heuristics)

| Setup | facts ok (answerable) | refused/declined (unanswerable) | wrongly refused (answerable) | citations valid | median latency |
|---|---|---|---|---|---|
| baseline-qwen0.5b | 13/27 | 0/3 | 0/27 | n/a (baseline has none) | 6.6s |
| final-qwen0.5b | 13/27 | 2/3 | 1/27 | 3/26 | 7.9s |
| final-llama8b | 23/27 | 3/3 | 1/27 | 26/26 | 1.6s |

Baseline relevance gate fired (fallback to general knowledge) on 0/30 questions.
