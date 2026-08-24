# ADR 0008: calibration coverage gates public ranking

## Outcome and non-goals

A released Compiler orders candidates for a profile only when that Compiler Revision's
`calibration_coverage` names the profile. Missing coverage is returned as missing rather
than projected as a precise order. This does not add a ranking term, predict time,
automatically promote calibration, or change the construction and verifier gates.

## Owners and dataflow

`compiler/revision.lock.json` is the sole authority for released calibration coverage.
`Compiler.assess` projects membership into `Assessment.calibration_available`, and
`Compiler.rank` replays the Assessment and consumes that fact before calling the
structural ranking primitive. The primitive remains independently testable, but it is not
a second authority and cannot make an uncovered profile covered.

For an uncovered profile, every otherwise lowering-eligible schedule is returned in the
withheld collection. A rejected schedule remains withheld for its gate findings. Thus the
caller retains the complete candidate set and can spend GPU time in provider order, while
never mistaking an uncalibrated heuristic for the paper's calibrated pre-GPU filter.

## Failure, compatibility and promotion

An Assessment from another Revision, or one whose recorded fields differ from canonical
replay, is refused. Frozen Compiler releases and their Study Contracts are unchanged; the
new behavior enters through a successor Compiler Revision. A profile may enter
`calibration_coverage` only in a later reviewed release whose declared evaluation domain
shows useful survivor selection under the fixed oracle and timing protocol.

## Acceptance evidence

The v7 B200 sweeps retain every raw sample for the complete declared GEMM and
Flash-KMeans domains. Flash-KMeans has 19.43% top-1 regret. GEMM has 1.72%, but one
non-preregistered sweep cannot define and pass its own promotion threshold, and the
hypothesis leaves performance-semantic ties unresolved. Neither supports released
coverage. Contract tests must show that public ranking withholds uncovered profiles while
the structural primitive remains deterministic. The full Corpus Gate must match before
releasing the successor Revision.
