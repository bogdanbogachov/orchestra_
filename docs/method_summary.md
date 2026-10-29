# Nuisance-subspace projection: method summary

2026-10-29

## Problem

The same engineering question, worded differently, can get different labels from an LLM classifier. Engineering has no room for that.

- Goal: the same answer for the same question, regardless of wording.
- Measured as group consistency (all duplicates get one label) and group all-correct rate (all get the right label).
- Constraint: accuracy must not drop by more than 0.5 points.

## Data

Real duplicates from the 2024-04-01 Stack Exchange dump (CC BY-SA), not synthetic noise.

- **Duplicate group:** questions that moderators linked as asking the same thing, 2 to 20 per group.
- **Singles:** questions with no duplicate link. Only these train the classifier.
- Groups are split 40/20/40 into fit / val / test; no group spans two splits.
- Leakage control: "Possible Duplicate" quotes and links to other questions are stripped.

| Task | Predict | Labels | Groups (fit / val / test) |
| --- | --- | --- | --- |
| site | Which site the question belongs to | 7 | 5174 / 2587 / 5175 |
| gis | Which GIS tool family | 7 | 918 / 459 / 918 |
| dba | Which database system | 5 | 513 / 256 / 515 |

## Method (geig)

Find the directions inside the model where duplicates differ but different questions don't, and remove them at inference. No training: one eigendecomposition and one matrix operation.

1. **Baseline:** fine-tune Llama-3.2-1B (LoRA + attention-pooling head) on singles, then freeze it.
2. **One vector per question:** at a chosen layer, average the question's word vectors (start token excluded) into one 2048-number vector.
3. **Noise arrows:** for each duplicate group, the arrow from the group centre to each member. Pooled over all fit groups, they give the within-group covariance Σ_w (2048 × 2048).
4. **Meaning arrows:** the arrow from the global centre to each group centre. Their covariance is Σ_b (2048 × 2048).
5. **Noise directions:** generalized eigenvectors that maximise noise spread ÷ meaning spread. Take the top k (the rank).
6. **Removal:** at the output of that layer, project every word vector (except the start token) off those k directions. The vector keeps its 2048 numbers; only the noise coordinates are reset to their average.

$$\Sigma_w u = \lambda (\Sigma_b + \varepsilon I) u, \qquad \varepsilon = \text{ridge} \cdot \operatorname{tr}(\Sigma_b) / d$$

$$h \leftarrow h - Q Q^\top (h - \mu), \qquad Q \in \mathbb{R}^{2048 \times k}$$

- Q holds the k noise directions as columns; μ is the average vector.
- The removal is linear, so projecting each word and then averaging equals averaging and then projecting. Directions fitted on question vectors therefore act exactly on word vectors.
- After removal, spread along those directions is zero at that layer. Later layers may partly recreate it; the pilot measures whether the output still improves.

## Baselines and pilot grid

geig has to beat three simpler ways of picking directions to remove.

| Method | Directions removed |
| --- | --- |
| geig (proposed) | Large noise spread ÷ meaning spread |
| sw | Large noise spread only (ignores meaning) |
| pca | Largest overall spread |
| random | Random directions |

- Positions: after decoder layers 4, 8 and 12 of 16, and after the final norm.
- Ranks k: 4, 16, 64, 256 directions out of 2048.
- Per task: 4 positions × 4 methods × 4 ranks = 64 settings on val, all inference only.
- Per method, keep the best all-correct rate with an accuracy drop of at most 0.5 points, then run it once on test with a group-level bootstrap.
- Cost: 3 training runs total (one per task), 3 parallel jobs, one GPU each.

## Gates and decision

Go to full scale only if A, B1 and B2 pass on at least 2 of the 3 tasks.

| Gate | Check | If it fails |
| --- | --- | --- |
| A | Baseline leaves ≥ 10% of val groups inconsistent | Task too easy; drop it. Says nothing about the method |
| B1 | geig gains ≥ 3 points in consistency or all-correct, accuracy drop ≤ 0.5 points | Hypothesis weak; stop |
| B2 | geig beats sw and random at the same layer and rank | Effect real but not novel; stop or rethink |
| B3 | A and B pass on ≥ 2 tasks | Inconclusive; no full-scale run |

If geig wins only at the final norm, the method reduces to post-hoc LDA on output features: a weaker paper.

## What would make it a paper

The pilot only shows whether there is a signal; reviewers will compare it to LDA, INLP and LEACE-style concept erasure.

- **Training-based baselines:** consistency-loss fine-tuning on duplicate pairs and paraphrase augmentation. Target claim: closed form, no retraining, matches or beats them.
- **Theory:** a bound showing the ratio criterion limits the logit change, explaining why geig keeps accuracy where sw loses it.
- **What the removed directions encode:** length, code blocks, error dumps, tone.
- **Breadth:** 2–3 model sizes or families, several seeds, cross-site transfer of the fitted directions.
- **Optional learned variant:** short fine-tuning with the projection in place.

Target venues: EAAI, ESWA or AEI. For ACL it would likely read as incremental.
