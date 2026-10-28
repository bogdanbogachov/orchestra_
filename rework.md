# Rework: from fixed FFT filtering to a new method

## Context

The current setup fine-tunes Llama-3.2-1B (LoRA) as an intent classifier on BANKING77 / CLINC150 with
synthetic conversational noise (prefix/suffix pleasantries, justifications, context, self-corrections,
typos in noise only). A fixed low-pass FFT over the sequence axis of last-layer hidden states, followed by
pooling (mean / max / last / attention), showed no meaningful improvement.

Why the sequence-axis FFT fails (diagnosis):

- The "FFT40" mask keeps 407/512 bins (~80% of the spectrum), so it only removes the top ~20% of
  frequencies; the filter is close to identity.
- It runs over the padded length (512), not the real length (~15-30 tokens): the cutoff is
  length-agnostic, and circular wrap-around leaks non-zero pad-position states into real tokens.
- Masked mean pooling is (approximately) the DC component, which a low-pass filter preserves; simulated
  mean-pooled cosine before/after filtering ≈ 0.9995.
- The injected noise is contiguous prefix/suffix phrases, i.e. *low*-frequency along the token axis, so a
  low-pass filter cannot separate it from content.
- After 16 layers of attention, noise is no longer localized to its own positions.
- The template noise is easy to ignore; LoRA + pretrained LM leaves little headroom.

Goal: a new method (not an applied study) suitable for a Q1 journal / top-tier conference.

---

## Idea A (recommended): Depth-Spectral Token Selection

**Hypothesis.** A token's hidden-state trajectory across layers is a signal over depth. Filler tokens
(pleasantries, closers, generic context) are "resolved" in early layers and then change little, while
task-relevant content tokens keep being transformed in deep layers. The spectrum / dynamics of the
per-token depth trajectory is therefore an unsupervised salience signal.

**Method.**
1. Run the backbone with `output_hidden_states=True`, giving `h_t^(l)` for each token `t` and layer `l`.
2. Per token, compute depth-trajectory features: FFT over the layer axis, or consecutive-layer deltas
   `||h^(l+1) - h^(l)||`, cosine drift, etc.
3. A tiny learned scorer maps these features to a per-token weight; pool with these weights
   (attention pooling driven by depth dynamics rather than final-layer content).
4. Green extension: compute salience at an intermediate layer `k` (e.g. 6/16) from layers `1..k`, drop
   low-salience tokens for layers `k+1..L` (valid for classification since nothing is generated),
   yielding robustness *and* FLOP/energy savings.

**Narrative.** Spectral analysis on the sequence axis provably fails (theory + current negative results);
spectral analysis on the depth axis works.

**Unique asset.** The noise generator knows exactly which tokens are injected noise, giving token-level
ground truth for analysis (evaluation only, never training).

**Go/no-go probe (~1 day, no training).**
- Extend `add_noise.py` to record character spans of injected fragments; map to token offsets.
- Run frozen Llama over noisy BANKING77, compute depth-trajectory features per token.
- Linear probe noise-vs-content token separability, evaluated on held-out (test) templates.
- Proceed if AUROC ≳ 0.85; abandon if near chance.

**Must-compare baselines.** Attention pooling, last-token, mean, layer-wise aggregation of hidden states,
LLMLingua-style perplexity-based token dropping, learned token pruning.

**Risks.** Prior work on layer dynamics / logit lens / early exit may already cover token salience from
depth; effect may not transfer to natural noise.

---

## Idea B: Mid-depth sparse token gating

**Method.** Learn hard-concrete (L0) gates on tokens at layer `k`; a sparsity penalty pushes the model to
keep a minimal sufficient token subset; gated tokens are dropped for subsequent layers. Trained
end-to-end with the classification loss.

**Pros.** Likely to work; directly yields robustness + compute savings.

**Cons.** Close to learned token pruning (LTP, PoWER-BERT) and rationale extraction (Lei et al. 2016,
Bastings et al. 2019). Novelty must come from the decoder-only SLM classifier setting and the joint
robustness/energy objective.

---

## Idea C: Noise-invariant pooling via paired consistency

**Method.** Train with paired clean/noisy views of the same query; add a consistency loss between their
pooled representations (and/or predictive distributions) alongside cross-entropy.

**Pros.** Almost certainly helps under clean-train / noisy-test shift; cheap to implement.

**Cons.** Consistency regularization (UDA, R-Drop, contrastive invariance) is well established; best used
as a component of A or B rather than the headline.

---

## Evaluation protocol (applies to all ideas)

- Clean-train / noisy-test distribution shift (in addition to noisy/noisy).
- Naturally noisy data, not only own templates. Note (lit review): MASSIVE en-US is SLURP text and HWU64
  is fairly clean crowdsourced text; only ASR transcripts of spoken corpora (SLURP, STOP) are genuinely
  noisy. Disfl-QA covers self-corrections (QA, not intent). Train-time and test-time noise families must be
  documented as disjoint for any "distribution shift" claim.
- Noise severity sweep as a controlled ablation.
- ≥3 backbones (e.g. Llama-3.2-1B/3B, Qwen2.5-0.5B/1.5B), 4–5 datasets.
- 10 matched seeds, paired tests with Holm correction (existing protocol).
- Report accuracy/F1, training energy, inference FLOPs/latency.

---

## Literature review (2026-10-27)

Three independent literature searches (arXiv, ACL Anthology, CVF, PMLR/ICLR/NeurIPS pages; 2016–2026).
Entries were verified at abstract level or better; those marked (FT) had method text read. Any citation
must be re-checked against the primary source before it goes into a `.bib`. Gaps in coverage: OpenReview
submissions under review, Google Scholar forward citations, Chinese-language venues.

**Own prior work, must be cited:** Bogachov, Letov, Zhao. *The Hidden States Cookbook: A Large-Scale
Ablation Study for Noise-Robust Conversational Intent Classification in Industry.* arXiv:2610.05394 (2026).
Any new paper must state its delta over this preprint; plan anonymisation if the venue is double-blind.
Its limitations section already points at "layer-aggregation strategies" and "learnable spectral
filtering", so Idea A reads as a natural follow-up.

### Verdict summary

| Idea | Verdict | Most threatening prior work | What remains defensible |
|---|---|---|---|
| A. Depth-spectral token selection | Partially covered | SBERT-WK (Wang & Kuo, TASLP 2020): cross-layer cosine-variance as token weight, "informative words vary more across layers" (FT). TransPrune (CVPR 2026): cross-layer transition scores for pruning, visual tokens only (FT). HSAD (arXiv 2509.13154): FFT over the layer axis, one token, hallucination detection (FT). | Per-token depth dynamics as salience in a **causal decoder classifier**; robustness to **additive conversational noise**; **text**-token pruning from layers 1..k; validation against known noise spans. |
| B. Mid-depth L0 token gating | Partially covered | ToP (Li et al., KDD 2023): hard-concrete/L0 token masks, removed from later layers, FLOPs constraint, encoder (FT). TokenGate (arXiv 2610.26754, Oct 2026): learned gate at a low layer of a decoder, upper layers skipped, top-K budget, generation only (FT). Chen et al. *Can Rationalization Improve Robustness?* (NAACL 2022): selection improves robustness to inserted text, prefix insertions hurt most (FT). | Content-conditioned L0 gates in a decoder **classifier**, joint robustness + **measured energy** objective. Each component has precedent. |
| C. Paired clean/noisy consistency | Partially covered, close to done as a mechanism | PPCL (Qiang et al., Findings EACL 2024): LLM intent classification, CE on both views + JS consistency (FT). CCL (Kim et al., NAACL 2024): representation + prediction consistency for ASR-robust SLU (FT). Activation Consistency Training (Irpan et al. 2025): activation invariance to added irrelevant prompt cues. | Only the setting: pooled representation of a small decoder, additive conversational noise, BANKING77/CLINC150. **Use as a component or baseline, not as the headline.** |

### Idea A: details

Closest work by role:

- **Premise ("filler settles early, content keeps changing"):**
  - SBERT-WK (encoder, training-free).
  - Conditional Scale Entropy (arXiv 2605.21391): wavelet over each token's depth trajectory separates metaphorical from literal tokens in decoders; analysis only.
  - FlexiDepth (arXiv 2503.23798): repetitive tokens and fixed phrases need fewer layers.
  - For decoders the premise is still only indirectly supported: a hypothesis to test, not a fact.
- **Depth-spectral features:** HSAD. Watch the wording collision with CLSE (arXiv 2606.24165), "spectral evolution-guided token pruning", which uses a *spatial* FFT on visual tokens. Always say "spectrum over the depth/layer axis".
- **Pruning from depth dynamics:** TransPrune, MSDG-Prune (arXiv 2609.36916), Token Tendencies (ACM MM 2026), all visual-token and training-free. Text-LLM prefill pruning: LazyLLM, FastV-style "score at layer k" (attention-based).
- **Layer-aware pooling for decoder classifiers:**
  - Token- and Layer-Selective Probes (Meyoyan & Del Corro, ACL 2026): token × layer probes on decoder LLMs incl. Llama-3.2-3B. **Closest classifier-side baseline.**
  - Layer-wise attention pooling (Oh et al., COLING 2022).
  - Layer by Layer (Skean et al., ICML 2025): intermediate layers often beat the final one.
- **Confound:** OrthoRank (ICML 2025): the sink/BOS token's state barely changes over depth, so it would look like "filler". Exclude it, and normalise per layer for norm growth.

Required refinements:

1. **Spectral must beat simpler alternatives.** It has to beat SBERT-WK variance weighting and raw consecutive-layer deltas. With 16 layers an rFFT has only 9 bins, so "spectral" may add nothing over variance. Test this first.
2. **Treat causality as the core difference.** Prefix filler is processed before the content exists, suffix filler after it. Report prefix and suffix noise separately.
3. **Separate the signal from the scorer.** Compare training-free depth weights vs learned scorer on depth features vs learned scorer on last-layer features (= current attention pooling). Don't call it "unsupervised" if the scorer is trained with labels.
4. **Pruning extension:** frame it as transferring transition-based scoring to text tokens with partial-depth salience. Report only measured FLOPs and energy.
5. **Use the known noise spans** (precision/recall of salience) as a validation that no prior work could run.

### Idea B: details

- **Rationalization line:** Lei 2016; Bastings 2019 (*HardKuma*, not hard concrete); DiffMask (De Cao et al., EMNLP 2020): hard-concrete gates predicted from layer-ℓ states, post hoc; Paranjape 2020 (IB); FRESH; SPECTRA; degeneration (Yu 2019, 2021).
- **Token pruning:** PoWER-BERT, LTP, LAT, TR-BERT, Transkimmer, ToP, Infor-Coef (claims first IB token pruning), Dynamic Context Pruning (NeurIPS 2023, decoder LM), Mixture-of-Depths, Router-Tuning.
- **Denoise + efficiency in LLMs:** END (arXiv 2502.18915): a linear probe on early layers drops noisy *chunks*, training-free.
- **Caveats:**
  - Kept tokens have already attended to the noise in layers 1..k (causal contamination).
  - Savings are bounded by roughly pruned fraction × (L−k)/L for short queries.
  - The pooling position must never be gated.
  - The gate pattern can leak the label.

### Idea C: details

- **Foundations/baselines:** UDA, R-Drop, SMART, SimCSE, Cheng et al. 2018 (adversarial stability for MT).
- **ASR-robust SLU:** SpokenCSE, ML-LMCL, MCLF, CCL.
- **Prompt invariance in LLMs:** CoIN, Flip-Flop Consistency (ACL 2026), BCT, ACT.
- **Augmentation-only controls:** Sengupta et al. 2021 (7 real-world noise types), Yoran et al. (ICLR 2024).

### Synthesis and recommendation

None of the three is a clean gap. The strongest defensible method **merges A and B**:

> **Depth-dynamics-conditioned token gating.** At layer *k*, a gate receives each token's depth-trajectory
> features from layers 1..k (deltas / drift / depth-spectrum, layer-normalised, sink excluded) and outputs a
> hard-concrete keep probability. An L0 penalty drives a minimal token set, and gated tokens are dropped for
> layers k+1..L of a causal decoder *classifier*. Optionally, Idea C's consistency loss is an auxiliary term.

How this differs from the threats:

| Prior work | What it does | Our difference |
|---|---|---|
| ToP | Attention-ranked masks, encoder | Gates conditioned on depth dynamics, decoder |
| TokenGate | Top-K budget, no L0, generation | L0-trained gates, classification |
| TransPrune | Training-free, visual tokens | Learned gates, text tokens, plus a robustness objective |
| SBERT-WK | Training-free pooling weights, encoder | Learned gating with pruning, causal decoder, robustness to noise |

The claim should be scoped: "to our knowledge, first to … in decoder-only classifiers under additive
conversational noise, with measured energy".

**Revised go/no-go probe (before any training).** On frozen Llama-3.2-1B over noisy BANKING77 with token-level
noise spans, compare the noise-vs-content AUROC of these features:

| Feature set | Role |
|---|---|
| (a) final-layer hidden state | Content-only baseline |
| (b) SBERT-WK cross-layer cosine variance | Prior-work baseline |
| (c) consecutive-layer deltas | Simple depth feature |
| (d) depth rFFT features | Proposed feature |
| (e) (a)+(d) | Combined |

Report all of them separately for prefix and suffix noise, and on held-out test templates. Then decide:

- **Proceed** only if depth features add clearly over (a) and (b).
- If (d) ≈ (c), drop the "spectral" framing and keep "depth dynamics".
- If depth adds nothing over (a), Idea A is dead. Fall back to B with content gates, framed around the
  decoder-classifier robustness + energy setting.

### Required baselines (union across ideas)

| Group | Baselines |
|---|---|
| Pooling | Last-token, mean, max, attention (from the Cookbook) |
| Layer-aware pooling | Best intermediate layer; learned layer mix / layer-wise attention pooling; token × layer probes |
| Training-free depth weighting | SBERT-WK adapted to decoders |
| Token pruning | Random drop at equal rate and layer; FastV-style attention pruning at layer k; LazyLLM; TransPrune TTV on text; ToP-style rank mask; stop-word/IDF drop |
| External compression | LLMLingua-2; Selective Context |
| Robustness training | Noise augmentation with CE only; PPCL-style JS consistency; R-Drop |
| Fair comparison | Matched-FLOPs comparison; energy measured with the same harness for all methods |

### Claims to avoid

- "First to use cross-layer dynamics for token importance" (SBERT-WK, TransPrune).
- "First FFT over transformer depth" (HSAD).
- "First hard-concrete / L0 token pruning" (ToP, DiffMask).
- "First IB token pruning" (Infor-Coef).
- "First consistency regularization for noise-robust intent classification" (PPCL, CCL).
- "Unsupervised salience" while the scorer is trained with labels.
- "MASSIVE / HWU64 are naturally noisy."
- Citing NoiseBench as an input-noise benchmark (it covers label noise in NER).
- "FFT does not help" in general. Only one fixed sequence-axis cutoff on final-layer states was tested.

### Open items

- Check forward citations of SBERT-WK, TransPrune, PPCL and ToP; this is the most likely place to find a
  text-domain follow-up.
- Search OpenReview (ICLR 2027 submissions).
- Reconcile noise-type wording between this repo and the preprint's Appendix A.
- Decide whether to cite the preprint openly or anonymise it.
- Confirm that the number of decoder layers in Llama-3.2-1B is 16, from its config.
