# Benchmarks: what is measured, against what, and how comparable it is

The README carries the headline numbers. This file carries the parts that do not fit
there: how the two tasks were framed, why the published results quoted beside them are or
are not on the same scale, and the full per-class breakdowns.

Every measured number here is reproduced from a file under `results/`, which
`scripts/collect_result.py` writes directly from a training run. Every published number is
transcribed from the paper named beside it and recorded with its table in
`results/published_baselines.json`. Nothing in this file is computed by hand.

## Why two tasks

Macro F1 over eleven sparse CWE classes was the only headline this repository used to
report, and it is a poor one. A macro average over classes with a few hundred training
rows each moves several points on the strength of one class, no published work on either
of these datasets evaluates that way, and a reader has nothing to compare it against. The
fix is not to drop it. It is to report it properly and to add a task that a reviewer can
actually place.

**Task A, top-10 CWE multiclass**, is the one the serving path uses: given a C or C++
function, which weakness class is it. It stays, now with per-class F1 beside the macro
average so a reader can see which classes carry it.

**Task B, binary vulnerable versus benign on CodeXGLUE Defect Detection**, is the standard
benchmark this project already downloads. It has a public leaderboard, a fixed split, and
a CodeBERT row on that leaderboard trained from the same checkpoint used here. It is the
number that says whether this pipeline works.

## How N was chosen for the top-N task

`TOP_K_CWES = 10` is set in `src/ml_inference/config.py` and was fixed before any model in
this repository was trained. It is the top ten CWEs by frequency among the CWE-labeled
vulnerable functions; everything else is collapsed into a single `__OTHER__` class.

`make cwe-support` prints the distribution the cut sits on and writes
`results/dataset/cwe_support.json`:

| Rank | CWE | Rows | Share of labeled rows |
|---|---|---|---|
| 1 | CWE-125 | 1635 | 10.2% |
| 2 | CWE-119 | 1433 | 8.9% |
| 3 | CWE-787 | 1379 | 8.6% |
| 4 | CWE-20 | 1314 | 8.2% |
| 5 | CWE-416 | 999 | 6.2% |
| 6 | CWE-476 | 915 | 5.7% |
| 7 | CWE-703 | 735 | 4.6% |
| 8 | CWE-200 | 716 | 4.4% |
| 9 | CWE-190 | 674 | 4.2% |
| 10 | CWE-399 | 462 | 2.9% |
| 11 | CWE-362, below the cut | 398 | 2.5% |
| 12 | CWE-400, below the cut | 395 | 2.5% |

There is no elbow at 10. Rank 10 holds 462 rows and rank 11 holds 398, and the curve keeps
decaying smoothly from there. Ten is a round number that keeps every named class above
roughly 450 rows, and the honest consequence is that the 132 CWEs below the cut are 36.3%
of the labeled data and `__OTHER__` is the largest class in the split.

That makes the accuracy column hard to read on its own, so the trivial predictor is
measured too. `make majority-baseline` always answers the training split's majority class,
which is `__OTHER__`, and scores **0.3764 accuracy, 0.0497 macro F1, 0.2059 weighted F1**
on the 2,415-row holdout (`results/majority_baseline/`). That is the floor every trained
result below has to clear on all three at once, and it is why an accuracy near 0.38 is not
by itself evidence that a model learned anything.

Choosing N after seeing scores would make the whole table meaningless, so the rule is
stated here and the distribution is committed. Anyone can rerun `make cwe-support` and
check that rank 10 is where this says it is.

## Comparability, stated plainly

| Published result | Same dataset | Same split | Same task | Directly comparable |
|---|---|---|---|---|
| CodeXGLUE Defect Detection leaderboard | yes | yes | yes | **yes** |
| Devign paper, Table 2 | partly | no | yes | no |
| DiverseVul paper, abstract and Table 6 | yes | no | no | no |

**CodeXGLUE Defect Detection** is directly comparable. This repository trains on the same
21,854 rows, evaluates on the same 2,732-row test split, and reports accuracy the same way.
The CodeBERT row on that leaderboard uses the same `microsoft/codebert-base` checkpoint
this repository fine-tunes. A gap against it is a gap in training budget and recipe, not
in what is being measured.

**The Devign paper** is the origin of the dataset but not of the split. Its Table 2
Combined column spans four projects (Linux Kernel, QEMU, Wireshark, FFmpeg) under a 75/25
train/validation split; CodeXGLUE releases only the FFmpeg and QEMU portion under an
80/10/10 split. Devign's 72.26% accuracy and 73.26% F1 on its own Combined set are listed
because a reader should know where the data came from, not because this repository is
being compared against them.

**The DiverseVul paper** never runs a multiclass CWE task. It reports binary detection over
the whole dataset, where roughly 94% of functions are benign, and its best model across
eleven architectures reaches 47.2% F1 at a 43.3% true positive rate and a 3.5% false
positive rate. Task A trains only on the 16,101 CWE-labeled vulnerable functions and asks
which weakness class a function is. No number in that paper sits on the same scale, and any
table that put them side by side without saying so would be misleading.

One result in that paper is worth reading next to Task A anyway, because it points the
other way. Table 6 reports CodeBERT on DiverseVul binary detection at 37.85% F1 with plain
cross-entropy and 41.72% F1 with class weights, at a small cost in accuracy (90.48% down to
89.39%). Class weighting helps there. On Task A it does not: the reweighted arm lands at
0.4048 macro F1 against the plain baseline's 0.4093, a tie. The two settings are not the
same problem. Their binary task is 94% benign, so upweighting the positive class buys
recall on the only class anyone cares about; Task A is an eleven-way split where the
reweighting moves error from the rare classes onto `__OTHER__` and the macro average
absorbs the trade. The accuracy cost points the same way in both: 90.48 to 89.39 there,
0.5350 to 0.4186 here.

## Sources

- Shuai Lu et al. **CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding
  and Generation.** NeurIPS 2021 Datasets and Benchmarks Track.
  [arXiv:2102.04664](https://arxiv.org/abs/2102.04664).
  Leaderboard: [Code-Code/Defect-detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).
- Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, Yang Liu. **Devign: Effective
  Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural
  Networks.** NeurIPS 2019.
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/49265d2447bc3bbfe9e76306ce40a31f-Paper.pdf).
- Yizheng Chen, Zhoujie Ding, Lamya Alowain, Xinyun Chen, David Wagner. **DiverseVul: A New
  Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection.** RAID
  2023. [Paper](https://surrealyz.github.io/files/pubs/raid23-diversevul.pdf),
  [arXiv:2304.00409](https://arxiv.org/abs/2304.00409).
- Zhangyin Feng et al. **CodeBERT: A Pre-Trained Model for Programming and Natural
  Languages.** Findings of EMNLP 2020. [arXiv:2002.08155](https://arxiv.org/abs/2002.08155).
- Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, Jian Yin. **UniXcoder: Unified
  Cross-Modal Pre-training for Code Representation.** ACL 2022.
  [arXiv:2203.03850](https://arxiv.org/abs/2203.03850).
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. **BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding.** NAACL-HLT 2019.
  [Paper](https://aclanthology.org/N19-1423/). Section A.3 is where the 2-to-4 epoch
  fine-tuning range used here comes from.

## Measured results, in full

Regenerate with `make results`, which reads only `results/` and computes nothing.
Every row traces to a `results/<run>/metrics.json` written by the training script itself.

## Task A: top-10 CWE multiclass on DiverseVul (11 classes with __OTHER__)

| Config | Train rows | Epochs | Macro F1 | Weighted F1 | Accuracy | Macro recall | Train wall clock |
|---|---|---|---|---|---|---|---|
| `majority class, no training` | 9854 | 0 | 0.0497 | 0.2059 | 0.3764 | 0.0909 | 0m 00s |
| `codebert-base, plain cross-entropy` | 9854 | 3 | 0.4093 | 0.5196 | 0.5350 | 0.3973 | 23m 31s |
| `unixcoder-base, focal + class weights + augmentation` | 10552 | 3 | 0.4048 | 0.4285 | 0.4186 | 0.4581 | 26m 46s |

### Per-class F1 on the holdout

| Class | Holdout support | Majority-class F1 | Baseline F1 | Improved F1 |
|---|---|---|---|---|
| __OTHER__ | 909 | 0.5469 | 0.6833 | 0.4453 |
| CWE-125 | 249 | 0.0000 | 0.5866 | 0.4969 |
| CWE-119 | 222 | 0.0000 | 0.4301 | 0.3993 |
| CWE-20 | 203 | 0.0000 | 0.2965 | 0.3676 |
| CWE-787 | 202 | 0.0000 | 0.5980 | 0.5471 |
| CWE-416 | 132 | 0.0000 | 0.4702 | 0.4302 |
| CWE-476 | 118 | 0.0000 | 0.3043 | 0.3333 |
| CWE-200 | 113 | 0.0000 | 0.2541 | 0.2762 |
| CWE-703 | 109 | 0.0000 | 0.2941 | 0.3433 |
| CWE-190 | 94 | 0.0000 | 0.4967 | 0.5251 |
| CWE-399 | 64 | 0.0000 | 0.0882 | 0.2888 |
| **macro average** | 2415 | 0.0497 | 0.4093 | 0.4048 |

## Task B: binary vulnerable/benign on CodeXGLUE Defect Detection (Devign)

| System | Accuracy | Binary F1 | Source |
|---|---|---|---|
| BiLSTM | 0.5937 | not reported | CodeXGLUE leaderboard |
| TextCNN | 0.6069 | not reported | CodeXGLUE leaderboard |
| RoBERTa | 0.6105 | not reported | CodeXGLUE leaderboard |
| CodeBERT | 0.6208 | not reported | CodeXGLUE leaderboard |
| **this repository, codebert-base, 2 epochs, 256 tokens, batch 16** | **0.6332** | **0.5197** | `results/devign_codebert/` |

## Published context

- DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection (RAID 2023 (26th International Symposium on Research in Attacks, Intrusions and Defenses)): best model over all 11 architectures F1 47.2; CodeBERT, no class weighting F1 37.85; CodeBERT, class weights on cross-entropy F1 41.72. The DiverseVul paper never runs a multi-class CWE task. It reports binary detection F1 over the whole dataset, where roughly 94 percent of functions are benign. The top-10 CWE task in this repository trains only on the 16101 CWE-labeled vulnerable functions and predicts which weakness class a function is, so no number in that paper sits on the same scale. Included because it is the published result on this dataset and it reports the class-weighting effect this repository also measures.
- Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks (NeurIPS 2019), Table 2, Combined column: Devign (Composite) accuracy 72.26, F1 73.26. Not the CodeXGLUE split. The Devign paper's Combined column spans four projects with a 75/25 split; CodeXGLUE releases only the FFmpeg+QEMU portion under an 80/10/10 split. Listed as the dataset's origin, not as a like-for-like comparison.
