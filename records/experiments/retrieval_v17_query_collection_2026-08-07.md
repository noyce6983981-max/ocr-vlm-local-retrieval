# V17 compositional query collection phase

## Outcome

- Extracted 162 public TextVQA source questions from the local manifest.
- Found 158 unique source questions.
- The deterministic bilingual V17 parser identified 56 unique compositional
  questions with at least two independently scorable visual requirements.
- Added 24 hard-negative wording proposals by changing exactly one color or
  spatial-relation condition.
- Produced 80 proposals: 40 calibration and 40 holdout.
- Kept source-neighbor groups within one split.
- Verified no normalized query overlap with 160 existing V16 or legacy blind
  queries.
- Did not run retrieval during query collection.
- A human reviewer completed all 80 wording decisions; 80 were approved and
  none required rejection or replacement.
- Froze the approved wording with query-set SHA-256
  `5950bd68bf7680aa186132d187ce65741d3b6f4245577a000c00314be8828979`.
- The freeze status is `query_wording_frozen_relevance_not_yet_judged`:
  wording approval must not be reported as relevance annotation.

## Provenance boundary

The 56 source questions were authored for TextVQA and retained verbatim. The 24
near-neighbor rows are system-generated proposals and cannot be called human
queries until a person approves or rewrites them. None of the 80 rows is a
relevance label or benchmark result yet.

## Artifacts

- `data/evaluation/v17/query_proposals/compositional_visual_proposals.jsonl`
- `data/evaluation/v17/query_proposals/query_review_queue.csv`
- `data/evaluation/v17/query_proposals/proposal_receipt.json`
- `scripts/v17_query_review_app.py`
- `scripts/freeze_v17_query_set.py`
- `data/evaluation/v17/frozen_query_set/frozen_queries.jsonl`
- `data/evaluation/v17/frozen_query_set/freeze_receipt.json`

The freeze command refuses incomplete reviews, duplicate wording, V16 query
overlap, non-compositional rewrites, rejected rows without replacements, or any
query-family group crossing calibration and holdout.
