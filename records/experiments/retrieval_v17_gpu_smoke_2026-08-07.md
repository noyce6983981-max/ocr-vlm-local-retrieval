# V17 compositional visual GPU smoke test

This is a motivating regression check, not a calibration or holdout metric.

- Query: `黄色海豚跃过沙漠金字塔`
- Policy: experimental V17 attribute coverage; thresholds are uncalibrated.
- Route: `visual_discovery`
- First candidate: `user_876860c6ee8a`, a pyramid near-neighbor.
- Global visual score: `0.538415`.
- Necessary-condition coverage: `1/6` (`16.7%`).
- Passed: pyramid object.
- Missing: yellow, desert, dolphin, yellow-dolphin binding, and dolphin-jumps-over-pyramid relation.
- Final decision: rejected.
- End-to-end runtime: `27.141 s` on the local RTX 4060 Laptop GPU environment.

The check supports the implementation claim that a high global similarity can
no longer override missing mandatory attributes. It does not support an
accuracy-improvement claim; that requires the frozen V17 human study.

## Parser V2 bilingual revalidation

After adding deterministic English term boundaries and TextVQA-compatible
relation markers, the same Chinese regression was run again to check that the
multilingual change did not alter its decomposition:

- Parser version: `2`.
- Requirements remained: yellow, desert, dolphin, pyramid, yellow-dolphin
  binding, and dolphin-jumps-over-pyramid relation.
- First candidate remained the same pyramid near-neighbor.
- Coverage remained `1/6`; final decision remained rejected.
- Runtime was `19.208 s` with text and BM25 caches warm and the visual attribute
  component recomputed for the new policy hash.
