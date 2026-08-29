# Labeling operations

How the active-learning loop routes an unlabeled pool, and how to convert its
output into a labeling-effort estimate using your own numbers.

## What the loop measures

`active_learning_loop.py` scores every row of `unlabeled_pool.parquet` with the
trained improved model and partitions it by the model's top-class confidence:

| Bucket | Confidence | Human action |
|---|---|---|
| Auto-labeled | >= 0.92 | Accepted as-is and folded into the next training run; spot-check a sample |
| Human review | 0.45 to 0.92 | Reviewed by a labeler |
| Uncertain | <= 0.45 | Prioritized for manual labeling, highest information gain |

Thresholds come from `AUTO_LABEL_THRESHOLD` and `LOW_CONF_THRESHOLD`, both
overridable from the environment.

The report at `bench/reports/active_learning.json` contains bucket counts,
`needs_human_fraction`, and `auto_label_accuracy_vs_truth`. The last of these
is a genuine check rather than an estimate: `build_splits.py` retains the pool's
real labels in a `true_label` column that the loop never reads before
predicting, so the auto-labels can be scored against ground truth after the
fact. On a real deployment that column does not exist, and the equivalent
guardrail is the spot-check sample.

## Converting a triage rate into hours

This project has not measured how long a human takes to CWE-label a C/C++
function, so it does not report hours saved. The arithmetic, if you have your
own per-function figure `t`:

```
hours_fully_manual = pool_size * t / 3600
hours_with_triage  = (human_review_queue + uncertain_queue) * t / 3600
```

The ratio between them is `needs_human_fraction`, which is the part the loop
actually measures. Two things make the hours figure much softer than the ratio:
reviewing a model's proposed label is faster than labeling from scratch, so the
review queue does not cost a full `t` per item, and the spot-check sample over
the auto-labeled bucket adds back cost the formula ignores.

## Quality guardrails

- Spot-check the auto-labeled bucket rather than trusting it. The auto-label
  bucket is only as good as the model that produced it.
- Track `auto_label_accuracy_vs_truth` between runs. A drop means the model
  drifted and the threshold needs raising.
- Raise `AUTO_LABEL_THRESHOLD` when agreement falls; that trades a smaller
  auto-labeled bucket for higher precision in it.
- Retrain before each loop iteration. Confidence from a stale model routes
  badly.

## Feeding results back into training

The loop writes auto-labeled rows to `data/splits/auto_labeled.parquet` with
columns `code`, `label`, `confidence`. `train_improved.py` loads that file when
it exists and concatenates it onto the training split before augmentation; set
`NO_AUTO_LABELS=1` to train without it. Because the loop needs a trained model
to score the pool, the order is: train, run the loop, then retrain to pick up
the new rows.
