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

- Query wording is frozen, but candidate relevance and answerability are not.
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
12 secondary. On the 12 double-reviewed queries, answerability exact agreement
is 91.7% (Cohen kappa 0.797); pooled candidate binary agreement is 95.8%
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

Pool-conditioned calibration results over 39 effective queries (23 answerable,
16 no-answer; `v17_visual_033` excluded for an unstable referent):

- V16 quality-hybrid Recall@3: 34.8%.
- V17 quality-hybrid Recall@3: 69.6%; paired difference +34.8 points, grouped
  Bootstrap 95% CI +8.7 to +59.1.
- V16 false-accept rate: 62.5%.
- Selected V17 false-accept rate: 31.3%, a 50% relative reduction; paired
  difference CI -60.0 to 0.0 points.
- V16 end-to-end Top-1: 23.1%.
- Selected V17 end-to-end Top-1: 51.3%; paired difference +28.2 points, CI
  +13.9 to +44.4.
- Selected V17 coverage: 43.6% (17/39 accepted), so the result is not produced
  by rejecting every query.

These are calibration results on model-assisted labels, not final benchmark
claims. The cross-encoder still confuses some `behind`/`in front of` and
`above`/`below` counterfactuals, and the false-accept interval reaches zero.
The 40-query holdout remains unopened and must stay sealed until an independent
human labeling plan and the final method lock are accepted.

Additional artifacts:

- `data/evaluation/v17/human_study/calibration/model_audit_report.json`
- `outputs/evaluation/v17/calibration/adjudicated_calibration_report.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_attribute_verification.json`
- `outputs/evaluation/v17/calibration/parser_v3/candidate_gate_calibration_report.json`
- `config/v17_candidate_verification_gate.json`
