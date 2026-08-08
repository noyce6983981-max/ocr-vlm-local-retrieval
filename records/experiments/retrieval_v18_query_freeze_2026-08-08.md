# V18 query-set freeze record

Date: 2026-08-08  
Study: `v18-condition-aware-listwise-160`  
Status: query wording frozen; retrieval not started

## Scope

V18 tests whether condition-aware, listwise best-or-none selection can prevent a
high-scoring attribute from hiding a missing mandatory condition in
compositional visual retrieval. The frozen set contains paired, source-grounded
Chinese queries: one positive and one single-condition hard negative for every
source group.

The first ten onboarding source groups were discarded and replaced under
amendment `V18-A002` before this freeze. Discarded rows remain outside the
effective benchmark as audit records.

## Human approval

- Active source groups: 80
- Approved source pairs: 80
- Reviewer ID: `query_author_01`
- Explicit single-condition confirmations: 80
- Residual machine-draft rows: 0
- Frozen queries: 160
- Globally unique normalized queries: 160

Machine-authored drafts were used only as editable starting points. A draft was
excluded from completion until the reviewer edited or checked it, explicitly
confirmed the one-condition constraint, and saved it as `approve`.

## Frozen balance

| Dimension | Count |
| --- | ---: |
| Calibration | 80 |
| Holdout | 80 |
| Positive | 80 |
| Single-condition hard negative | 80 |
| Attribute-object binding | 40 |
| Spatial relation | 40 |
| OCR plus visual condition | 40 |
| Multi-condition scene | 40 |
| Chinese | 160 |

## Integrity hashes

| Artifact | SHA-256 |
| --- | --- |
| Effective V18 protocol | `3f3ea13e57bf31b28d9281fb445b74d7d1f724c75342c01508ad8e0951714d70` |
| Authoring queue | `f94ec8803bbbe4dc4948fe319947a07e5e33f22b299d72ad00860212f3e10a5c` |
| Approved submissions | `870be24e66c56067998c20135918430cdc0ae1e1591b9f5df476f8c5d01da314` |
| Frozen query set | `1f62947074d5fe67374a25c4453900b0ca665b1d2e6110da094ec1481de5e897` |

The private freeze receipt records `retrieval_executed=false`,
`holdout_results_opened=false`, and `v17_artifacts_modified=false`.

## Governance after freeze

- Query wording, source bindings, split assignments, query roles, and query-set
  hash are immutable for this study.
- Method selection and threshold calibration may use only the calibration split.
- The holdout split remains sealed until the final method and runtime manifest
  are locked.
- Any post-lock method, prompt, threshold, dependency, or query-set change
  requires a new study rather than a V18 holdout rerun.

## Next stage

Implement and test the four protocol methods (`L0`-`L3`), generate calibration
candidate pools, obtain blinded candidate relevance judgments, calibrate on the
calibration split, and lock the complete runtime manifest before the single
authorized holdout run.
