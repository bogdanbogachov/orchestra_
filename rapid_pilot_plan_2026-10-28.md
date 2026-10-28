Rapid research decision plan — 28 October 2026 (revised: nuisance-subspace projection)

This revision replaces the earlier trajectory-informed token-pruning plan. A backup of the previous
version is kept outside the repo; the earlier direction is superseded, not merged.

## 1. Problem

**Engineering problem: prediction consistency.** Engineers ask the same question in different
words: abbreviations, typos, back-story, different order. A router or classifier must give equivalent
inputs the same output.

- Argmax classification is already deterministic per input. The failure is *inconsistency across
  equivalent inputs*, and accuracy hides it.
- Primary metric: **group consistency**, the fraction of equivalence groups whose members all receive
  the same prediction.

**Hypothesis.** Hidden states of a fine-tuned causal decoder classifier contain a low-rank *nuisance
subspace*: directions that vary across equivalent phrasings far more than across different questions.
Projecting it out of the residual stream at an intermediate layer, for every token, raises group
consistency without reducing accuracy.

Motivation (architectural argument, not an established fact): in a causal decoder, token pruning
cannot remove nuisance already absorbed by retained tokens, whereas a subspace projection acts on every
token. "Prune directions, not tokens." The FFT negative result motivates this: it filtered along the
sequence axis, which is not where nuisance lives.

## 2. Method

At layer k of the LoRA-fine-tuned classifier:

1. Pool residual-stream states per input over real tokens (mean, consistent with the head).
2. From training-split duplicate groups, estimate:
   - Σ_w: within-group covariance (members minus group mean);
   - Σ_b: between-group covariance (group means minus global mean).
3. Solve the regularized generalized eigenproblem Σ_w u = λ (Σ_b + εI) u. Take the top-r eigenvectors
   as nuisance subspace U, orthonormalized.
4. Insert P = I − UUᵀ as a forward hook on decoder layer k output, applied to every token, then
   complete the forward pass.

Variants:

| Variant | Description |
|---|---|
| Training-free | Projection inserted into the already fine-tuned model |
| P + LoRA re-tune | Projection fixed, LoRA and head re-tuned on top of it |
| Final-only | Projection only on the final pooled feature (post-hoc LDA/WCCN-style); placement control |

**Key ablation.** Grant & Wang (arXiv 2610.17117) show that ablating paraphrase-variable directions
estimated from Σ_w alone destroys fact recall. The method's claim rests on the between-group
normalization avoiding that failure. Compare:

| Ablation | Purpose |
|---|---|
| Σ_w-only top-r removal (NAP/GRIT style) | Tests the between-group normalization |
| Top-r PCA removal | Variance-only control |
| Random rank-r subspace | Chance control |

Each is swept over r ∈ {4, 16, 64, 256} and k ∈ {4, 8, 12, final}. Confirm the layer count from the
model config before running.

## 3. Novelty position (literature check, 28 Oct 2026)

**Verdict: partially covered.** No paper found does the full combination:

- nuisance subspace estimated from **naturally occurring duplicate groups**;
- within- vs between-group generalized eigenproblem;
- projection **inside a causal decoder's residual stream** at an intermediate layer, all tokens;
- evaluated on **classification group consistency + accuracy**.

Every ingredient has precedent:

| Prior work | Overlap |
|---|---|
| Dasgupta, Tanvir & Zhong (arXiv 2605.06458) | **Most threatening.** Same estimator on decoder LLM states, used for model attribution, with no projection-out and no classification |
| Grant & Wang (arXiv 2610.17117) | Within-pair covariance subspaces; removal harms recall |
| Zhao et al. (arXiv 2505.15257) | Inference-time projection of a nuisance (language) subspace from LLM hidden states |
| NAP / WCCN (speaker verification, 2005–2007); GRIT (arXiv 2505.24843, vision) | Projecting out within-class or within-pair variability |
| LEACE, INLP, SAL, SPLINCE | Concept erasure (needs a labelled concept) |
| ReFT/LoReFT | Learned residual-stream interventions |
| ACT, F2C, PPCL, CoIN | Consistency training |

**Defensible claim.** Transfer of NAP/LDA-style nuisance projection into an LLM residual stream for
classification consistency, supervised by natural duplicate groups, with evidence that between-group
normalization preserves accuracy where Σ_w-only removal does not. **Not** a new subspace estimator.
This suits EAAI/AEI; it is not a top-tier methods claim.

Claims to avoid:

- First paraphrase/nuisance subspace in LLMs.
- First generalized-eigen semantic/paraphrase separation.
- First residual-stream projection.
- First within-class nuisance removal.
- First paraphrase-consistency metric.
- "Removes nuisance from every token": only the linear component along U at layer k onward is removed.
- "Token pruning cannot remove absorbed noise" as a fact: it is a hypothesis.

Re-run the novelty search immediately before submission; this geometry is active in 2026 preprints.

## 4. Data (verified 28 Oct 2026)

**Source and licence.** Official Stack Exchange dump, **2024-04-01 snapshot** on archive.org
(`archive.org/download/stackexchange/`).

- Licence is CC BY-SA, with the per-post version in `ContentLicense`.
- The snapshot predates the 2024 click-through restricting LLM training. Do **not** use newer gated
  dumps for fine-tuning without contacting Stack Exchange.
- Attribution: site, per-question URL, author. Any released derived dataset must be ShareAlike.
- Filter watermark rows (IDs ≥ 1e9).

**Measured statistics** (questions only; groups = connected components of duplicate links, size ≥ 2):

| Site | Groups | Within-group dof | Median words (p90) | Label |
|---|---:|---:|---|---|
| unix | 4,884 | 9,887 | 117 (320) | Coarse family (0.80 dup agreement) or site routing |
| gis | 3,070 | 4,360 | 115 (274) | 7 tool/platform families (0.95 agreement, chance 0.38, coverage 0.81) |
| electronics | 2,129 | 3,142 | 119 (291) | Site routing; tags too noisy (0.59 top-20 agreement) |
| dba | 1,475 | 2,181 | 144 (384) | 7 DBMS families (0.97 agreement, chance 0.46, coverage 0.92) |
| diy, mechanics, networkengineering, engineering | 920 / 303 / 205 / 56 | – | ~110 | Site routing only |

**Tasks.**

| Task | Description |
|---|---|
| T1, primary, engineering framing | **Site routing** over electronics, unix, dba, gis, diy, mechanics, networkengineering (~13k groups). Labels are perfectly consistent within duplicate groups by construction |
| T2 | gis family routing (within-site) |
| T3 | dba DBMS-family routing (within-site) |

Fine-grained tags are not used as labels: duplicates share the top tag only 40–80% of the time.

**Data rules.**

- Cap hub groups at ≤ 20 members (keeps >99% of groups).
- Text = title + HTML-stripped body; truncate at 512 tokens.
- Use dynamic padding, not padding to `max_length`.
- Measure token lengths with the Llama tokenizer before fixing truncation.
- Splits are **by group**: no group spans train/val/test.
- Non-duplicate questions supply additional classifier training data.
- Σ_w and Σ_b are fitted only on train-split groups. r and k are selected on val groups. Test groups
  are used once.
- Family label rules (`coarse.py` regexes) must be reviewed by hand before use.

## 5. Model and setup

- Llama-3.2-1B with the existing LoRA configuration and pipeline.
- Restore `downloaded_models/downloaded_3_2_1b` first; it is currently absent.
- Heads: attention pooling (the Cookbook's best) and last-token.
- Extend `commands/training/data.py` loaders to the new JSON format.
- Add a layer-k projection hook to `models/` (forward hook on `model.layers[k]`).
- Keep the matched-seed protocol and paired tests with Holm correction.

## 6. Metrics

| Metric | Definition |
|---|---|
| Accuracy, macro-F1 | Over all test questions |
| Group consistency | All members receive the same prediction |
| **All-correct group rate** | All members correct. Consistency alone rewards being consistently wrong |
| **Discrimination** | Accuracy on test questions from different groups and labels, plus between-group separation of pooled states. Collapsing everything gives perfect consistency |
| Calibration | ECE, optional |

Report the projection's effect on each metric with paired, group-level bootstrap CIs.

## 7. Schedule and decision gates

**Day 1 — build and baseline.**

- Extract T1–T3 from the 2024-04 dumps. The scripts already exist in the scratchpad from the data check.
- Train baseline LoRA classifiers (attention + last-token, 1 seed).
- Record runtime; earlier BANKING77 runs took ~21–25 min, and longer texts will take more.

**Gate A — headroom.** The baseline must leave room to improve: at least ~10% of multi-member test
groups inconsistent on T1 or T2. If consistency is already ~98%, the problem is too easy on this data.
Then switch to the harder label (gis/dba families, or top-20 tags as a noisy-label condition) before
proceeding.

**Day 2 — training-free pilot.**

- Fit Σ_w and Σ_b at k ∈ {4, 8, 12, final}.
- Insert the projection with the r sweep; evaluate on val groups.
- Run Σ_w-only, PCA and random-subspace ablations.

**Gate B — go/no-go.** Proceed only if all three hold on val:

1. Generalized-eigen projection raises group consistency or all-correct rate by a practically
   meaningful margin (target ≥ 3 points absolute) at ≤ 0.5 points accuracy loss.
2. It beats the random subspace and the Σ_w-only removal at matched r.
3. The effect survives on a second task.

If only the final-layer variant works, the in-stream contribution collapses to post-hoc LDA: stop and
reconsider. If nothing beats random, stop.

**Days 3–5 — full experiments.**

- 3 tasks × 2 heads × 3–5 seeds.
- Proposed variants: training-free, P + LoRA re-tune.
- Ablations: Σ_w-only, PCA, random, layer/rank sweeps, final-only.

Baselines:

| Group | Baselines |
|---|---|
| Training controls | LoRA alone; LoRA with duplicate-group augmentation at the same data budget |
| Consistency training | Output-level pairwise JS/KL (PPCL/F2C-style); activation-level L2 at layer k (ACT-style) |
| Matched-budget intervention | LoReFT |
| Erasure | LEACE/SPLINCE with explicit Z = duplicate vs non-duplicate pair |
| Encoders | SBERT-linear (repo baseline); one strong encoder (ModernBERT or DeBERTa) |

Measure training and inference energy with the existing harness.

**Days 6–7 — manuscript.**

- Target: EAAI, then AEI.
- Reuse the Cookbook's pooling results, energy method and statistics, citing it as prior work.
- Framing: consistent routing of engineering questions with small LLMs via residual-stream nuisance
  projection.

## 8. Compute (unmeasured planning estimates)

- One H100/A100 is ample; 24 GB VRAM suffices for the 1B model.
- Covariance fitting is cheap: one forward pass over train groups per layer, and an eigenproblem in
  2048 dimensions.
- The bulk is LoRA training: about 3 tasks × 2 heads × 4–5 conditions × 3–5 seeds.
- Benchmark one run on day 1 and multiply out before committing to the full grid; cut seeds to 3 if
  needed.

## 9. Risks

- **Low headroom.** Site routing may be near-trivial (Gate A).
- **Accuracy damage.** The projection may remove label-relevant directions (Grant & Wang); the
  between-group normalization must be shown to prevent this.
- **Scooping.** The novelty margin is narrow and preprints are appearing quickly; re-check before
  submission.
- **Duplicate semantics.** Some duplicates share a root cause rather than being paraphrases; define
  equivalence as "moderator-marked".
- **Licence.** Use only the 2024-04 snapshot.
- **Engineering framing.** unix, dba and gis are IT/geospatial; the electronics, mechanics and
  networkengineering sites carry the engineering framing in T1.
