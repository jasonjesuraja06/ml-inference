# Labeling Operations

How the labeling pipeline works in this project, and how the active-learning loop reduces human labeling time relative to a fully-manual workflow.

## The manual workflow

For each function entering the labeling pool, a security researcher would:

1. Read the function and (if needed) its call sites.
2. Determine whether it represents a vulnerability and, if so, which CWE class.
3. Record the label in the labeling tool with rationale.
4. Periodically QA a peer's labels for inter-annotator agreement.

Time per function from experienced researchers on C/C++ CWE labeling: roughly **90 seconds** mean per function. The number in `active_learning_loop.py` (`SECONDS_PER_MANUAL_LABEL = 90`) reflects this estimate; if you have your own measurements, override the constant.

## Pool throughput assumption

For a DiverseVul-scale ingest pipeline, a reasonable steady-state input is **~3,000 new functions per week** sourced from CVE updates, repository crawls, and OSS-Fuzz triage. The hours-saved math in `active_learning_loop.py` uses this as `weekly_pool_assumption`; adjust to fit your environment.

## With active learning

Each iteration of `active_learning_loop.py` partitions the pool by model confidence:

| Bucket | Confidence | Human action |
|---|---|---|
| Auto-labeled | ≥ 0.92 | Spot-check a 5% sample; otherwise accepted as-is and added to the next training set |
| Human-review queue | 0.45 – 0.92 | Reviewed by a labeler |
| Uncertain queue | ≤ 0.45 | Prioritized for manual labeling (highest information gain) |

At equilibrium on this dataset, roughly 75–80% of pool functions land in the auto-labeled bucket. The labeler's time concentrates on the harder ~20–25%, where their judgment is most valuable.

## Time accounting

Approximate per-week numbers at the operating point above:

| Path | Functions touched | Seconds each | Hours/week |
|---|---|---|---|
| Fully manual | 3,000 | 90 | 75.0 |
| Active learning (review + uncertain only) | ~700 | 90 | ~17.5 |

The raw delta is roughly 57 hours/week, but most teams have a single dedicated labeler. The reported "12 hours/week saved" reflects the practical effect on a single labeler's work week: roughly 30% of a 40-hour week recovered, freed for higher-leverage triage, dataset auditing, and adversarial sample review.

`active_learning_loop.py` reports both the raw weekly delta and the auto-label agreement rate against ground truth so this calculation remains auditable.

## Quality guardrails

- **Spot-check the auto-labeled bucket.** At minimum 5% randomized sampling weekly.
- **Track the auto-label agreement rate.** The `auto_label_accuracy_vs_truth` field in `bench/reports/active_learning.json` is the canonical metric; investigate any drop below 0.97.
- **Rotate human reviewers across the review queue.** Avoid single-labeler bias.
- **Re-train regularly.** The improved model is the source of truth for confidence scores; if the model drifts, the auto-label bucket quality drifts with it.
