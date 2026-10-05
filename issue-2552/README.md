# Issue #2552: B200 validation data

Recorded measurements for [issue #2552](https://github.com/Dao-AILab/flash-attention/issues/2552) and the independent-width approach in [PR #2780](https://github.com/Dao-AILab/flash-attention/pull/2780).

## Revision and provenance

- Baseline: upstream [3451a2a](https://github.com/Dao-AILab/flash-attention/commit/3451a2a67a24eeed8af54d3c5b8d219577113797).
- Patched source: an uncommitted local snapshot; exact kernel and test SHA256 values are in [provenance.json](provenance.json). These identify the files used for the measurements, independently of a later repair-branch commit.
- Measurement date: 2026-10-05. The date is the experiment date, not the asset publication date.
- This branch stores evidence only. It does not contain the kernel patch or experiment harnesses.

## Data index

| Evidence | Files |
| --- | --- |
| Per-attempt correctness, including O/dQ/dK/dV errors and compilation failures | [JSONL](correctness.jsonl), [summary CSV](correctness_summary.csv) |
| Complete backward medians for 14 controls | [CSV](benchmark.csv), [JSON](benchmark.json) |
| Seven alternating rounds per control (98 pairs) | [rounds CSV](benchmark_rounds.csv) |
| Test counts and cache-only execution results | [test results](test_results.json), [logs](logs/) |
| Actual generated MHA disassembly | [baseline](sass/baseline.txt), [patched](sass/patched.txt), [comparison](sass/comparison.diff), [load counts](sass/load_counts.txt) |
| Source, original log and CUBIN hashes; environment | [provenance](provenance.json) |

Published logs replace local repository and Python environment path prefixes. Original log hashes are retained in provenance.json; the published files have their own hashes in [manifest.json](manifest.json).

## Correctness protocol

BF16, causal, Dqk=128, Dv=96, seed 0. The focused configurations use B=1, Sq=129, Sk=257, Hq=4, Hkv=4/2. The reported-shape configurations use B=2, Sq=Sk=1024, Hq=16, Hkv=16/8. The reported-shape label refers to dimensions from the issue, not identical reference arithmetic or error values from its author.

Each revision runs in a fresh process with persistent FA4 caching disabled and CUTE_DSL_NO_CACHE=1. Within a process, successfully compiled kernels are reused across 10 calls with the same inputs. Failed GQA compilation is retried on each attempt; these are not 10 independent successfully compiled binary samples.

Maximum absolute errors are against the repository attention_ref with FP64 inputs. Acceptance follows check_tensor_vs_ref using a low-precision reordered reference and a rounding allowance of 2 * dtype epsilon * maximum reference magnitude. O, dQ, dK and dV are checked. The new runtime tests also check finite outputs and mean error; deterministic cases require bitwise-repeatable gradients.

The neutral control retains the original width and storage semantics while adding independent attributes and compile-time branch structure. Both focused failures remain in 10/10 attempts.

## Test coverage

74 new tests passed FakeTensor compilation. Real execution passed 48 main-matrix and 26 edge cases, with zero kernel compilations in both selections. Another 24 existing dense/varlen backward cases passed in a single pass: 98 GPU correctness cases total.

Main matrix: FP16/BF16, Dv=32/96, MHA/GQA/MQA, dense/packed-varlen, causal/noncausal. Edge cases: input Dv=24/80, forced 1CTA, deterministic GQA, and head-dimension controls. Input Dv=24 is padded to 32; input Dv=80 remains 80.

Kernel Ruff check and format check passed. The existing test file has 25 pre-existing Ruff diagnostics; the new tests add none. git diff --check passed for the repair diff. The FA2 extension was missing; a process-local FA4 namespace launcher isolated it without changing the FA2 entry point.

## Performance protocol

BF16, B=2, Sq=Sk=2048, Hq=8. Each timing includes complete backward: preprocess, main kernel and postprocess. Forward and compilation are excluded.

Baseline and patched use separate compilation caches and CUDA Graphs on the same GPU. Each graph contains 10 complete backward calls. Before capture, each variant executes three calls. After capture, 30 alternating replay pairs warm up both graphs. Each timed sample replays one graph 20 times (200 backward calls). Seven rounds alternate which revision is measured first. CUDA events measure elapsed GPU time; reported values are per-backward medians across rounds.

The measured median latency change range is -0.7853754742107588% to +1.1648999296888007%. No listed control exceeds the 5% threshold. GPU clocks were not locked; small changes can reflect noise. Incorrect asymmetric baseline outputs are excluded from performance comparisons. This does not establish behavior outside the listed workloads, on SM110, or across the full parameterized suite.

## Environment

NVIDIA B200 / SM100; driver 580.126.20; Python 3.12.14; PyTorch 2.13.0+cu130; CUDA 13.0; CUTLASS DSL 4.7.1; apache-tvm-ffi 0.1.11; quack-kernels 0.6.5.
