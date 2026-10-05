# PR 2941 supporting assets

Static figures, original timing data, correctness logs, and provenance for the
local CLC L2 swizzle repair. This branch contains assets only, without product
code, test code, or a PR implementation diff.

- Baseline: `3451a2a67a24eeed8af54d3c5b8d219577113797`.
- Local patch: `b0986d7105356c1d67f692af0292dd4abaeae443`.
- Original optimization: [PR #2941](https://github.com/Dao-AILab/flash-attention/pull/2941).
- Main STATIC 2CTA support: [PR #2962](https://github.com/Dao-AILab/flash-attention/pull/2962).

## Figures

- `figures/clc_swizzle_speedup_distribution.png` and `.svg`: all 180 shapes.
- `figures/clc_swizzle_representative_latency.png` and `.svg`: selected gains and
  the largest regression, with all three round means shown.

Image captions, environment, methods, and limitations belong in the PR body.
Use a commit-pinned raw.githubusercontent.com URL when embedding a PNG, so the
PR figure stays tied to its data even if this branch changes.

## Data

| Path | Content |
| --- | --- |
| data/benchmark_measurements.csv | 1,080 original timing rows |
| data/benchmark_per_shape.json | Original per-shape results and three rounds per revision |
| data/benchmark_per_shape.csv | Derived 180-row medians, speedups, and round values |
| data/benchmark_summary.json | Original aggregate metrics |
| data/benchmark_environment.json | Toolchain, scheduler hashes, and configuration evidence |
| data/verified_metrics.json | Independently recalculated metrics and test counts |
| data/correctness_evidence.jsonl | 76 O/LSE observations across 68 test names; source log and line included |
| data/logs/ | Raw test/benchmark logs, sanitizer controls, and unresolved-report ledger |
| data/upstream/ | GitHub API snapshots and related-issue search results |
| data/provenance.json | Revisions, source hashes, and claim-to-evidence mapping |
| manifest.json | SHA256/size of all loose asset files |
| pr2941_assets.zip | Figures and evidence together with the manifest |

The O/LSE JSONL uses `logs/` source paths for this assets layout. Published logs replace local usernames, repository paths, Python environment
paths, and user-specific cache paths with placeholders. Numerical benchmark
data is unchanged; original hashes for redacted logs are retained in provenance.json. No source snapshots or
executable scripts are included in this branch.

## Results and scope

332 unique tests passed with two existing skips. The original 32 O/LSE tests
are included in the expanded 68. The 180-shape benchmark contains 1,080 timings:
geometric-mean speedup 1.03938x; 138 lower/42 higher median latencies; maximum
median latency increase 3.52586%.

Clocks were unlocked and some GPU 1 timings overlapped tests on GPU 0. Latency
changes do not establish statistical significance or an L2 hit-rate mechanism.
Coordinate probes cover supplied 2CTA CLC responses; full attention coverage
is 1CTA hardware CLC plus the STATIC hd256 2CTA fallback. All 180 benchmark O
outputs matched main bitwise, separate from independent-reference tests.
Device memcheck passed five cases with API reporting disabled; the earlier
unfiltered run retains 41 unresolved CUDA API query reports.
