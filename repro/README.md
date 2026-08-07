# 40-page minimal reproduction package

This package creates a deterministic 40-page synthetic document library. The
pages, text, labels, and layouts are generated locally by this repository and
released under CC0-1.0. No private corpus, model weight, or network download is
needed for the CPU path.

## CPU path

From the repository root:

```powershell
python scripts/minimal_repro.py ingest
python scripts/minimal_repro.py evaluate --backend cpu
```

The first command generates 40 PNG pages, a page manifest, a known-text corpus,
a BM25 index, 16 natural-language queries, and `expected_results.json` under
`artifacts/minimal_repro/`. The second command evaluates answerable and hard
no-answer queries. The expected aggregate output is committed as
`repro/expected_results.json`.

The CPU path is a retrieval fallback and deliberately uses the generator's
known source text. It verifies corpus generation, indexing, ranking, exact
necessary-condition rejection, and evaluation; it does not claim to reproduce
PaddleOCR or VLM quality without those model environments.

## Optional GPU visual path

After installing the documented Qwen3-VL environment and model weights:

```powershell
python scripts/minimal_repro.py ingest --with-visual-index
.\.venv-vl\Scripts\python.exe scripts/minimal_repro.py evaluate --backend visual
```

The visual result is reported separately and is not required to equal the CPU
expected JSON. V17 attribute thresholds remain an uncalibrated research
scaffold until the human calibration/holdout protocol is completed.
