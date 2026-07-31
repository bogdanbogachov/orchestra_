## Reviewer Comment 1

Thank you for your review!
1. **Industrial evaluation.** The CFP states that proprietary data need not be released; the submission follows that rule. We also ran an anonymized offline evaluation on real procurement interactions. Labeled real data is scarce: the reliable subset supports five deployed classes, not all 200 templates. It contains **607 queries** over `order_search`, `late_parts`, `part_line_items`, `part_search`, and `general_search` (**486/121 train/test**). This is not presented as a full 200-way replacement test; it is a high-support deployed slice where supervised evaluation is possible. Sanitized example: "status of order [PO_ID]" -> `order_search`.

2. **Incumbent comparison.** The evaluated component is query/SQL-template selection; SQL execution and final LLM summarization are identical after selection. On the five-class real selector slice, the incumbent Titan+Claude proxy reaches **81.05%** strict response success, while Llama attention reaches **92.23 ± 2.14 acc / 92.20 ± 2.19 F1** and FFT40+attention reaches **92.64 ± 1.67 acc / 92.57 ± 1.71 F1**. The five-way task is easier than the full 200-way library, so the operational claim is a phased selector replacement for high-volume intents with fallback, not blanket proof for all templates.

3. **Cost claim.** The savings argument is not based on untested accuracy equivalence. On the evaluated real-log selector slice, the classifier is cheaper and more accurate than the Titan+Claude selector proxy. The cost claim is scoped to the selector stage and the covered high-support intents.

4. **Headline gain and significance.** The **+2.6-2.8 F1** headline is the gain over default head C0. Against the fairer custom last-token C3, paired tests over 10 matched seeds give **+0.34 F1** on noisy BANKING77 (`p=0.027`, Holm `p=0.108`) and **+0.47 F1** on noisy CLINC150 (`p=0.156`, Holm `p=0.312`), so proxy pooling-only gains are modest and not significant after correction. On the five-class real slice, attention beats C3 by **+8.82 F1** (`p<0.001`, Holm `p<0.001`) and C0 by **+11.43 F1** (`p<0.001`, Holm `p<0.001`). Across many matched-seed runs, this pattern was consistent. The abstract and conclusion will therefore avoid claiming universal "attention for noisy"; the defensible claim is that attention is a strong green/Pareto pooling choice, while last-token remains competitive.

5. **Energy/carbon framing.** The green result is a training-energy/convergence result, not lower per-step pooling cost. Attention vs mean uses fewer steps and less energy on BANKING77 noisy (**1823 ± 124** vs **2505 ± 117**, **0.27 ± 0.02** vs **0.38 ± 0.02 kWh**) and CLINC150 noisy (**1973 ± 129** vs **3025 ± 296**, **0.36 ± 0.02** vs **0.55 ± 0.06 kWh**). This matters because the selector may be retrained frequently, potentially weekly, as user questions evolve.

6. **Noise protocol.** The noise experiment measures representative randomized production-like noise, not arbitrary unseen distribution shift. The six mechanisms are inspired by observed user behavior in the real industrial logs. Train/test instances are random; the six mechanisms are shared, while train/test use separate style/template pools. The same noisy files are used across all methods, so the generator does not selectively favor attention.

7. **Deployment scope.** The deployed assistant is typed, and observed users generally ask one operational question at a time, so one selected SQL/query intent is the normal path. ASR is outside the interface; code-switching, ASR, and multi-intent turns are future stress tests rather than central current workload.

8. **FFT protocol.** The same noisy BANKING77 split was used for the FFT cutoff sweep and final reporting, so fixed FFT is treated as exploratory. Preliminary post-submission adaptive-cutoff checks did not change the conclusion. Clean BANKING77 FFT-mean is noted as an outlier: **82.17 ± 2.30 F1** vs **88.65 ± 0.54** for non-FFT mean, consistent with fixed spectral truncation removing useful clean-signal information.

9. **CLINC/OOS and split detail.** CLINC OOS examples were included as a regular **151st label** in training and evaluation. A clean-train/noisy-test condition would be a stronger distribution-shift stress test; the current noisy setting is deliberately a familiar but randomized noise-family test.

## Reviewer Comment 2

Thank you for your comments.
1. **Backbone generality and green pooling.** The target is cost-sensitive industrial deployment with small language models, hence the deliberate focus on Llama-3.2-1B-Instruct. Prior work supports broader pooling relevance across transformer variants (Ennadir et al., 2025), while empirical claims from the original grid are for this SLM. The added DistilBERT runs provide cross-backbone evidence for the core green-pooling point: attention uses less training energy than CLS across all matched datasets, while model choice and accuracy remain deployment-dependent. The abstract and conclusion will make this claim explicit.

2. **Industrial/end-to-end evaluation.** The real-log evaluation targets the deployed bottleneck: query/SQL-template selection. Downstream database retrieval and final LLM summarization are shared by both systems. On the anonymized five-intent real subset, the incumbent Titan+Claude selector proxy reaches **81.05%**, while the best Llama configuration reaches **92.64 ± 1.67 acc / 92.57 ± 1.71 F1**. This supports a scoped high-volume selector replacement with fallback, not a full 200-template proof.

3. **Proxy datasets vs procurement queries.** BANKING77 and CLINC150 match the task structure, not the domain vocabulary: short user utterance -> one intent. The real subset shows the same structure: median query length **6 words**, **473/607** queries are <=8 words, and many are direct operational requests with IDs/entities. The real procurement evaluation addresses the domain-vocabulary gap.

4. **Intent/noise representativeness.** Real logs support the synthetic noise design. Observed typed usage includes short questions, incomplete references, IDs/part/order numbers, casing/punctuation variation, typos, abbreviations, and supplier/part wording. The dataset is small enough for manual audit, and the six noise mechanisms reflect these observations. The claim is representative of observed typed procurement usage, not ASR/code-switching or arbitrary distribution shift.

Reference: Ennadir et al. (2025), *Pool Me Wisely: On the Effect of Pooling in Transformer-Based Models*, NeurIPS. https://arxiv.org/abs/2510.03339

## Reviewer Comment 3

Thank you for your comments.
1. **Backbone transferability and green pooling.** The original controlled grid fixes one cost-sensitive decoder-only SLM, so empirical claims from that grid are backbone-specific. Broader relevance is grounded in prior pooling work (Ennadir et al., 2025). A targeted DistilBERT check shows the same sustainability pattern: attention trains with less energy than CLS on every matched dataset: real **0.0003 vs 0.0009 kWh**, BANKING77 clean/noisy **0.0043/0.0042 vs 0.0047/0.0047**, CLINC150 clean/noisy **0.0064/0.0063 vs 0.0067/0.0068**. Accuracy remains dataset-dependent, but the training-energy advantage is consistent.

2. **Lightweight baselines.** Real five-intent results are: SBERT-linear **95.84 F1**, DistilBERT-attention **95.49 ± 1.35**, DistilBERT-CLS **94.67 ± 0.81**, best Llama **92.57 ± 1.71**; DistilBERT-attention is also faster (**3.47 ms/query**) than Llama attention (**15.50 ms/query**). Thus the paper should not claim that Llama-1B+attention is the best accuracy/latency choice. The abstract/conclusion will state the core claim: greener attention pooling across model families; backbone choice is deployment-dependent.

3. **Energy mechanism.** Training energy is central because the selector may be retrained frequently, potentially weekly, as user questions evolve. The differences are mainly convergence-speed effects. On noisy BANKING77, attention uses **1823 ± 124 steps / 1492 ± 95 s / 0.27 ± 0.02 kWh** vs mean **2505 ± 117 / 2104 ± 94 s / 0.38 ± 0.02 kWh**; on noisy CLINC150, **1973 ± 129 / 1947 ± 127 s / 0.36 ± 0.02 kWh** vs **3025 ± 296 / 2983 ± 297 s / 0.55 ± 0.06 kWh**. Inference remains low: real-set Llama attention **15.50 ± 0.44 ms/query**, **0.0006 kWh** test-set inference; DistilBERT-attention **3.47 ± 0.02 ms/query**, energy below table precision.

4. **Statistical tests.** Paired tests over 10 matched seeds show attention vs custom last-token gives **+0.34 F1** on noisy BANKING77 (`p=0.027`, Holm `p=0.108`, dz `0.83`) and **+0.47 F1** on noisy CLINC150 (`p=0.156`, Holm `p=0.312`, dz `0.49`), so proxy pooling-only gains are not significant after correction. On the five-class real slice, attention beats last-token by **+8.82 F1** (`p=0.000103`, Holm `p=0.000824`, dz `2.08`), but this is not a full 200-way selector proof. We will add p-values, CIs, and effect sizes.

5. **Noise generation.** Each query receives one noise fragment with probability **0.70** or two with **0.30**; fragment type is uniform over six categories. Self-correction is **0.18 train / 0.12 test**, typo probability on injected fragments **0.20 / 0.10**, and length growth is capped at **45%**. Added-token means are BANKING77 train/test **+3.82/+4.23** and CLINC150 **+2.56/+3.19**. Train/test use separate style pools, and all methods use the same noisy files, so the generator does not selectively favor attention.

6. **Novelty and FFT details.** The paper is not only an ablation of known pooling operators: it tests 1D FFT low-pass filtering of SLM hidden-state sequences before pooling as a noise-removal step for intent classification, which, to our knowledge, has not been evaluated before. The negative FFT result is therefore part of the contribution. Inputs are padded to `max_length`; FFT uses a fixed 40% cutoff over the full hidden-state sequence, with no length-normalized FFT, windowing, or learnable spectral filter. Padding/length artefacts may confound FFT, so fixed FFT is treated as exploratory.

7. **CLINC150 OOS.** OOS examples are included as a regular **151st class**; reported F1 is closed-set multiclass F1 including OOS. We will also add one-vs-rest OOS AUROC, FPR@95%TPR using OOS-class probability, and OOS precision/recall/F1. Calibrated rejection thresholds remain future work.

8. **Attention-weight analysis.** We analyzed learned attention-pooling weights on 200 noisy BANKING77 examples. For 184 comparable cases, **88.54%** of pooling mass falls on original query tokens vs **11.41%** on injected noise; **81.09%** of top-5 weighted tokens come from the original span. These pooling-head weights are not causal explanations, but they substantiate that the head often down-weights greetings/fillers and concentrates on intent content.

9. **Scope clarifications.** The attention head is a single learned token-weighting layer; multi-head/additive/gated pooling are future variants. We will clean figure/caption wording and replace "controlled distribution shift" with "mild lexical/style shift."

Reference: Ennadir et al. (2025), *Pool Me Wisely*, NeurIPS. https://arxiv.org/abs/2510.03339
