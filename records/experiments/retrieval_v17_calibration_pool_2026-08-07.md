# V17 calibration candidate-pool build

## Outcome

- Froze 80 human-approved query wordings with SHA-256
  `5950bd68bf7680aa186132d187ce65741d3b6f4245577a000c00314be8828979`.
- Executed retrieval for the 40-query calibration split only.
- Did not execute any of the 40 holdout queries.
- Scored 40 global visual queries plus 116 visual attribute requirements in one
  Qwen3-VL model load over 1,482 library items.
- Materialized independently versioned text, BM25, global visual, V16 hybrid,
  V17 hybrid, and V17 attribute-coverage rankings.
- Initially built the full blinded union of each run's Top-20 candidates:
  3,115 query-candidate pairs, with 63 to 89 candidates per query.
- After reviewer feedback, archived that pool and replaced it with exactly 20
  candidates per query: the union of all six runs' Top-3 candidates first,
  followed by RRF fillers. The active pool has 800 query-candidate pairs.
- Migrated all 6 already-completed judgments without requiring re-review. Two
  previously identified relevant items fall outside the capped pool and remain
  explicitly recorded in migration metadata.
- Assigned all 40 queries to the primary reviewer and a deterministic 12-query
  subset to the secondary reviewer for 30% double annotation.

## Correctness finding

The first audit found that the mixed-route ranking policy replaced the V17
attribute-aware hybrid ranking with the V16 `quality_rrf` ranking. The attribute
scores were present but their fused order was discarded. The route policy now
preserves the V17 quality-hybrid order whenever attribute coverage is active,
and a regression test covers this case.

After the fix, V16 and V17 have different run hashes. Their Top-20 overlap is
5 to 14 items per query (median 8.5), and 31 of 40 Top-1 candidates differ.
These are pipeline sanity checks only; they are not performance claims because
human relevance judgments do not yet exist.

## Audit boundary

- Query wording is frozen. The 20-item task judges pooled candidate relevance,
  not corpus answerability.
- Reviewer packets contain no run IDs, scores, ranks, acceptance decisions, or
  expected answerability.
- The holdout remains sealed until the parser, prompts, thresholds,
  aggregation, configuration, and code revision are locked.
- The capped pool supports complete per-run Top-1/Top-3 judging. It does not
  support unbiased Recall@10, Recall@20, or nDCG@10 claims; those endpoints were
  removed from the active calibration protocol.
- Pre-cap artifacts and judgments remain recoverable under
  `data/evaluation/v17/human_study/calibration/archive/`.

## Reviewer reset

The reviewer reported that the active judgments were based on a mistaken
reading and requested a full progress reset. Ten active judgments were removed
from the study on 2026-08-07 and must not be used in any metric or agreement
calculation. A recoverable copy is retained only for audit under
`archive/user_reset_20260807/invalidated_judgments_before_reset.jsonl` with
SHA-256 `65b22cd5bb2e7b572644583b28ec5b97dee6c929560da6b878cac37b524a95a7`.

A second reviewer-requested reset invalidated two additional judgments after an
ambiguous query-selector UI was reported. Those judgments are excluded from all
metrics and archived under `archive/user_reset_20260807_round2/` with SHA-256
`bcee8bbef7415c972eea72eb73759b6d005ef51481a9abf7f65b98e96f4c36bf`.
The selector now shows an explicit current-binding card containing both the
active query ID and wording, uses a fingerprint-scoped state key, and refuses a
query-ID/packet mismatch.

## Artifacts

- `outputs/evaluation/v17/calibration/calibration_retrieval_receipt.json`
- `outputs/evaluation/v17/calibration/runs/`
- `data/evaluation/v17/human_study/calibration/pool_audit.jsonl`
- `data/evaluation/v17/human_study/calibration/review_packets.jsonl`
- `scripts/v17_relevance_review_app.py`

## Completed review and model-assisted audit

The active review now contains 52 preserved human judgments: 40 primary and
12 secondary. On the 12 double-reviewed queries, the legacy answerability
field—now interpreted only as pooled relevance—has exact agreement of 91.7%
(Cohen kappa 0.797); pooled candidate binary agreement is 95.8%
(kappa 0.645). These figures describe reviewer agreement, not retrieval
accuracy.

A separate visual audit found systematic necessary-condition errors in the
human rows, including treating a logo *on* a shirt as a logo *under* a shirt,
the zone behind a hitter as the zone in front, a side advertisement as a rear
advertisement, and arbitrary numerals as price evidence. The 52 human rows were
not overwritten. Instead, `adjudicated_judgments.jsonl` records a distinct
model-assisted calibration layer with its provenance and file hashes. It must
not be described as human gold.

## Parser-v3 and candidate-verification result

The original parser incorrectly emitted object requirements such as
`what letter is` and `what kind of zone is`. Parser v3 rewrites visual-QA
answer slots into observable evidence, for example `visible letter under his
cap`, `zone in front of the hitter`, and `white visible word on the engine`.
All 40 calibration queries were rescored in one Qwen3-VL embedding load; the
original V17 raw artifacts remain untouched under the parent calibration
directory, while parser-v3 artifacts live under `parser_v3/`.

Strict single-vector attribute coverage remained invalid: it rejected all 39
effective queries, and its weakest-attribute score had Top-1 relevance AUC
0.38. Threshold relaxation was therefore abandoned instead of tuning the same
broken evidence scale.

The B5 follow-up uses Qwen3-VL-Reranker-2B as a candidate-level verifier. It
scores the full query and every necessary condition without reading judgments.
On calibration, geometric aggregation over condition scores was selected at a
0.45 threshold using a fixed 0.01 grid, an explicit false-accept reduction
constraint, and a reject-all prohibition.

Pool-conditioned calibration results over 39 effective queries (23 with a
relevant candidate in the 20-item pool, 16 with no relevant candidate in that
pool; `v17_visual_033` excluded for an unstable referent):

- V16 quality-hybrid Recall@3: 34.8%.
- V17 quality-hybrid Recall@3: 69.6%; paired difference +34.8 points, grouped
  Bootstrap 95% CI +8.7 to +59.1.
- V16 pool-conditioned false-accept rate: 62.5%.
- Selected V17 pool-conditioned false-accept rate: 31.3%, a 50% relative
  reduction; paired
  difference CI -60.0 to 0.0 points.
- V16 end-to-end Top-1: 23.1%.
- Selected V17 end-to-end Top-1: 51.3%; paired difference +28.2 points, CI
  +13.9 to +44.4.
- Selected V17 coverage: 43.6% (17/39 accepted), so the result is not produced
  by rejecting every query.

These are 20-pool calibration diagnostics on model-assisted labels, not
corpus-level open-set results or final benchmark claims. The cross-encoder
still confuses some `behind`/`in front of` and
`above`/`below` counterfactuals, and the false-accept interval reaches zero.
The 40-query holdout remains unopened and must stay sealed until an independent
human labeling plan and the final method lock are accepted.

Additional artifacts:

- `data/evaluation/v17/human_study/calibration/model_audit_report.json`
- `outputs/evaluation/v17/calibration/adjudicated_calibration_report.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_attribute_verification.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_gate_calibration_report.json`
- `config/v17_candidate_verification_gate.json`

## Top-K, contrastive-relation, OCR, and parser follow-up

The original 20-item pool contained every V17 Top-3 candidate but missed seven
items from ranks 4-5. Those seven images were reviewed separately for
calibration and stored in append-only extension files; the original packets
and judgments were not changed. The extension is model-assisted and is not
human gold.

One complete K=5 verifier artifact then supplied the common scores used to
derive K=1, K=3, and K=5. Every operating point returned the highest original
retrieval rank that passed, so a higher verifier score could not reorder two
passing candidates. The grid also compared exact positive-vs-inverse relation
margins and deterministic OCR precedence. The fixed selection rule retained
operating points within 0.03 absolute accuracy of the best result, then chose
the smallest K.

Selected calibration candidate over the same 39 effective queries:

- K=3, full-query verifier score, threshold 0.51.
- Original-rank-first among passing candidates.
- Pool-conditioned end-to-end accuracy: 56.4% (old K=1 candidate: 51.3%).
- Pool-conditioned false-accept rate: 25.0% (old K=1 candidate: 31.3%).
- Relevant-query success: 43.5% (10/23), one more correct relevant query than
  the K=1 candidate.
- Acceptance coverage: 48.7%; mean candidates inspected under the sequential
  decision rule: 2.10.
- V16-to-selected end-to-end paired difference: +33.3 points, grouped
  Bootstrap 95% CI +18.9 to +50.0.
- V16-to-selected false-accept paired difference: -37.5 points, grouped
  Bootstrap 95% CI -62.5 to -13.3.

The contrastive-relation branch was not selected. At its best eligible points
it reduced false acceptance to 18.8%, but the hard relation condition also
reduced relevant-query success. K=5 did not improve the best accuracy beyond
K=3 and required more candidate checks. No calibration query contained an
explicit target OCR phrase, so the implemented exact/fuzzy OCR precedence has
no effect estimate in this experiment.

The independent parser diagnostic used a 40-query Codex-assisted calibration
reference, explicitly marked as non-human-gold. Requirement precision, recall,
and F1 were 86.5%, 79.2%, and 82.7%; directional-relation query accuracy was
77.5%, and attribute-binding accuracy was 75.0%. Nine queries exposed nested
relation or binding granularity errors that should not be attributed to the
retrieval model.

The K=5 GPU scorer used a resumable checkpoint. One forced interruption after
query 1 was resumed at query 2; the combined artifact reports 118.88 seconds
and 4.02 GiB peak reserved VRAM. Because candidates were batch-scored, this is
not a sequential production latency measurement; inspected-candidate count is
only a cost proxy.

Governance and reproducibility:

- `data/evaluation/v17/amendments/004_topk_calibration_selects_k3_full_query.json`
- `data/evaluation/v17/human_study/calibration/top5_extension_review_packets.jsonl`
- `data/evaluation/v17/human_study/calibration/top5_extension_judgments.jsonl`
- `data/evaluation/v17/parser/calibration_parser_reference.jsonl`
- `config/v17_candidate_verification_gate_topk_candidate.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_attribute_verification_top5_contrastive.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_gate_topk_calibration_report.json`
- `outputs/evaluation/v17/calibration/parser_v3/parser_requirement_metrics.json`

These remain pool-conditioned calibration diagnostics. Corpus answerability
has not received two-reviewer adjudication, and the 40-query holdout remains
sealed.

## Final method lock and one-shot final-evaluation guard

The selected K=3 method is now frozen in `config/v17_method_lock.json`. The
lock binds the calibration-selected source commit, nine method files, three
governance files, the exact verification prompts, the frozen query-set
fingerprint and file bytes, parser and ranking fingerprints, five private
runtime artifacts, all 16 local Qwen reranker files, and the inference and
aggregation settings. The model downloader did not persist a remote revision;
this limitation is explicit rather than replaced with a guessed revision. The
complete 4,271,051,812-byte local snapshot is instead bound by manifest SHA-256
`af310f8b0b664b4ca85d486b1614f3b2909104dbb046535e5d258d5aa74705b8`.

`scripts/run_v17_holdout_once.py` defaults to read-only preflight. Execution
requires an explicit `--execute` plus an authorization document proving two
complete independent blind reviews, no model-assisted labels, independent
conflict adjudication, and exact hashes for the verifier output, adjudicated
judgments, and frozen V16 baseline. It atomically creates an exclusive receipt
before invoking the evaluator. A receipt blocks every retry, including after a
failed evaluator process, and the output is also create-only.

The guard has deliberately not been authorized or executed. No receipt or V17
holdout result exists, and V16 remains the latest independent public result.
Governance is recorded in
`data/evaluation/v17/amendments/005_method_locked_holdout_execution_guarded.json`.
The initial lock commit failed the isolated strict-Ruff step before activation;
the formatting-only correction and revision-2 lock are appended, without
rewriting A005, in
`data/evaluation/v17/amendments/006_pre_activation_ci_format_normalization.json`.
