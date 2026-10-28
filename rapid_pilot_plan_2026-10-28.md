Rapid research decision plan — 28 October 2026

**Objective and uncertainty.** Decide within 24–48 hours of a working GPU environment whether partial token trajectories reveal a reproducible failure of existing pruning scores and help avoid it under conversational nuisance changes. A successful pilot warrants further experiments; it does not certify novelty or top-tier acceptance. A week of experiments followed by 2–3 days of drafting is an aggressive, conditional schedule.

The motivating literature and its limitations are recorded in `literature_review_2026-10-28.md`. The candidate contribution is a specific, replicated failure mechanism and an effective response to it. Adding trajectory features alone is not sufficient.

**Resources already present.** The repository contains clean and noisy BANKING77 LoRA adapters, tokenizer artifacts, and saved classifier heads:

- `experiments/63/banking77_clean_custom_attention_7/custom_head`
- `experiments/62/banking77_noise_custom_attention_7/custom_head`

Their saved training logs report approximately 21 and 25 minutes for the earlier training jobs, but those figures do not predict the runtime of the new experiment. The configured base-model directory `downloaded_models/downloaded_3_2_1b` is absent locally. Restore the exact compatible checkpoint before reusing adapters. Existing job scripts request an H100 80GB; actual access and queue times remain unconfirmed. Local GPU inventory could not be verified because NVML access was blocked. No new experiments were run while preparing this plan.

**Public data.**

| Resource | Use | Source |
|---|---|---|
| BANKING77 | Initial pilot; fine-grained intent distinctions | [Original release](https://github.com/PolyAI-LDN/task-specific-datasets) |
| CLINC150 | Confirm the result in additional domains; use in-scope labels for the core experiment | [Original release](https://github.com/clinc/oos-eval) |
| SNIPS with independently released noise test sets | External robustness evaluation independent of our template generator; use intent labels only | [Noise benchmark](https://github.com/amazon-science/real-world-noisy-benchmarks-for-natural-language-understanding) |
| HWU64 | Optional further intent dataset if runtime allows | [DialoGLUE download/preprocessing scripts](https://github.com/alexa/dialoglue) |
| Disfl-QA | Optional task-transfer study only; this is QA and requires a different evaluation setup | [Original release](https://github.com/google-research-datasets/disfl-qa) |

The released noise suite covers phenomena such as misspellings, paraphrases, and abbreviations. These are not interchangeable with additive conversational filler. Evaluate each family separately and scope claims accordingly. Public synthetic perturbations are acceptable controlled experiments, but cannot establish production effectiveness by themselves. All data versions, source licences, splits, and transformation rules should be recorded.

**The hypothesis to test.** At a fixed pruning depth and token budget, current-state or attention gates sometimes discard needed information after a harmless conversational change. Early representation trajectories help a small gate predict and prevent this harm. This is a hypothesis, not an observed failure. Later computation is determined by the complete current hidden state; our claim is about useful predictive features for limited-cost gates.

**Stage 0 — working setup and implementation checks, first few hours.**

Restore the base model and reproduce saved classifier predictions on a small development sample. Confirm tokenizer, padding, checkpoint and pooling compatibility. Add a single intermediate pruning hook with correct causal attention, position IDs and readout handling. Keeping all tokens must reproduce dense logits within numerical tolerance. Account for every retained token in subsequent tensor dimensions. Disable dropout for interventions. Check both irregular masks and consecutive removals. These checks are prerequisites: positional or masking bugs can mimic the hypothesized phenomenon.

Use the clean checkpoint for primary discovery and the noisy checkpoint as a stress control. If the supposed failure disappears after ordinary noise training, that is relevant evidence against a substantial method claim.

**Stage 1 — inexpensive discovery, day 1.**

Start with approximately 400–600 stratified development queries from BANKING77. Recover split provenance first. Do not use the official test set to choose features, depths, thresholds or transformations. Split predictor fitting, tuning and pilot confirmation by source query, with all variants/interventions from a query kept together. A held-out noise family must also be absent from fitting/tuning. Existing model training exposure and development-set reuse should be disclosed; final confirmation uses the prescribed held-out test protocol.

Create a clean input plus three noise variants per source: a greeting/polite prefix, a suffix, and a fluent, label-preserving addition with distracting content. Audit a sample from every family for label preservation. Keep the task fixed: an insertion that adds another genuine request creates ambiguity rather than valid nuisance. Randomize placement and vocabulary independently of intent. Keep natural lengths and use several mild severities; do not inflate inputs solely to manufacture speedups.

Use three predefined depths, initially 4, 8 and 12 for a 16-layer checkpoint. Cache states at these depths in a single dense forward pass. For each input and depth, sample roughly six removal sets spanning single word/subword groups and multiple-token sets. Include random and score-based sets. Use exact injected-span removals only as a separate oracle diagnostic, not as gate supervision. Include fixed practical retention budgets such as 75% and 50%.

Replay only the layers after the intervention. Batch compatible replays. This is on the order of tens of thousands of short suffix evaluations, rather than training a new language model. Use an initial measured microbenchmark to decide whether the workload fits available GPU hours. Do not store every full hidden state for every layer if streamed trajectory summaries suffice.

For each removal, measure supervised cross-entropy change, correct-class margin change, and whether a previously correct prediction became wrong. Report harmful and beneficial removals separately. Dense-prediction disagreement alone is insufficient: pruning can correct an existing mistake. Single-token effects do not establish the safety of deleting a group.

**Stage 2 — does history add operational value, day 2?**

Fit small predictors of removal harm with identical splits, tuning and comparable capacity:

1. Current token representations with a cheap pooled view of the retained/removed context, set size and position information.
2. The same inputs plus early cosine drift and relative-L2 trajectories.
3. Current-state inputs plus shuffled or random history features, controlling extra dimensions and parameters.
4. Attention and faithfully adapted SBERT-WK/Representation Shift scores as simple controls.

Use at least a linear predictor and a small MLP so that a deliberately weak baseline does not manufacture a history advantage. Include current-state norm, position and tokenization-length controls. Compare global nuisance effects to pruning-specific damage relative to each input's dense result.

Evaluate on held-out source queries and unseen noise families. Use loss prediction/ranking as screening metrics, then make actual pruning choices and execute them. A better AUROC or regression score is not sufficient unless it yields better classification at the same retention budget. Candidate-set selection is an offline feasibility diagnostic; any eventual deployment method must account for selection overhead and cannot use multiple expensive suffix trials at inference.

Inspect a prespecified random sample of failures as well as cases where methods disagree. Test the resulting explanation with a targeted intervention, such as moving the same filler, preserving/removing the alleged evidence span, or changing pruning depth. At least one clean replication should be reserved after formulating the explanation. Plots of scores alone cannot establish a mechanism.

**Decision at 48 hours.** Proposed triage targets below are project choices, not journal requirements. Fix them before inspecting confirmation data:

- A reproducible pruning failure exists at a useful budget, beyond ordinary dense-model errors under noise.
- Trajectories improve held-out harm prediction by a practically noticeable amount; for example, roughly 10% relative reduction in prediction error. This is supporting evidence only.
- Actual decisions reduce pruning-induced classification errors relative to a strong current-state gate at matched retention. As a prioritization target, look for an absolute improvement around 1–2 percentage points in affected conditions, with uncertainty reported. Do not promote a result resting on a handful of flips.
- Effects survive capacity, context, position, norm and tokenization controls and remain visible with the noise-trained checkpoint. A promising discovery on one checkpoint needs independent confirmation before the main claim.
- Keeping all tokens validates correctly; physical compaction shows a credible runtime benefit on representative input lengths. An indicative practical target is at least 10% net latency reduction at the intended quality level. If only artificially long inputs benefit, scope or reject the systems claim.

If the sample has too few harmful events or wide confidence intervals, the result is inconclusive. Increase the independent confirmation sample once, without changing the hypothesis or mining the test set. If the effect disappears, stop this history-based method direction. Statistical significance alone is insufficient; bootstrap at source-query level because variants and removals are correlated.

**Full experiment schedule, conditional on pilot success.**

| Day | Deliverable |
|---|---|
| 1 | Restore checkpoint, validate pruning, run intervention discovery |
| 2 | Matched predictors, unseen-noise confirmation, targeted mechanism check, runtime feasibility; go/no-go |
| 3 | Freeze method and benchmark protocol; complete baseline port; run representative training/timing jobs |
| 4–5 | BANKING77, CLINC150 and SNIPS; two model families; three matched seeds; launch the focused method/baseline grid |
| 6 | Replication and ablations; external noise suite; small human-verified intent-changing contrast set; real latency and energy |
| 7 | Complete remaining runs, paired uncertainty estimates, failure analysis, artifact audit and evidence table |
| 8–10 | Draft and revise with the user, contingent on results supporting the claims |

The minimal full grid uses two small decoder families, three public intent tasks and three matched seeds. Four trained conditions per setting would produce 72 configurations before ablations. Shared frozen features/backbones can reduce some cost, but such experiments must be described accurately and supplemented by fair trained baselines. Attention, random and other training-free scoring conditions mainly add inference runs. Benchmark a representative job first and multiply by the actual planned configuration count. GPU queues, implementation difficulties and independent replication may extend the schedule.

Required trained comparisons include a strong dense classifier, dense robustness training, a PRUNEPERT-style current-state gate receiving the same robustness training, and the trajectory-enhanced gate. Include at least one strong small encoder classifier to assess the decoder's efficiency case. All retention methods use consistent pooling and comparable training resources. Confirm current-state-gate robustness without trajectories as the primary ablation. Reproduce prior methods faithfully where possible and label adaptations explicitly.

For the intent-changing contrast set, create a small audited subset of public examples with corrected labels. A negation or entity replacement does not necessarily change the intent taxonomy; verify the new label. Report pair correctness, not merely whether predictions differ. Automated generation may propose examples, but the generator's label is not ground truth without validation. Private industry data is unnecessary.

Primary paper outputs: a reproducible failure map across noise and depth; a controlled explanation of the failure; a method ablation establishing the contribution of trajectories; accuracy-versus-latency/energy curves with real compaction; and generalization to an independent model, public domain and noise source. Keep the official tests out of method selection.

**Writing and venue scope.** Three writing roles can work after the evidence is stable: methods/experiments writer, introduction/related-work writer, and skeptical reviewer who checks claims against results and sources. They should use one results manifest and one lead integrator. A 2–3-day draft is plausible when analysis is finished, references and figures are ready, and the user's feedback is available; revision readiness and acceptance cannot be scheduled in advance.

ACL/EMNLP is the closest target for a convincing language-specific discovery. ICLR requires a broader adaptive-computation result; intent-only feature improvements are a weaker fit. Advanced Engineering Informatics still requires an engineering contribution, even if data are public. Treat top-tier quality as an evidence threshold and allow a stop decision; a deadline cannot establish novelty or experimental success.
