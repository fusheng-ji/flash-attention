# Memcheck API reports during CuTe initialization

Toolchain: B200, CUDA 13.0, PyTorch 2.13.0+cu130, CUTLASS DSL 4.7.1.

Observed: `exhaustion_memcheck.log` contains 41 cuGetProcAddress_v2
CUDA_ERROR_INVALID_VALUE API reports and 5 passing exhaustion tests. All 41
reports precede the pytest result; no device Invalid read/write reports appear.
These API reports are kept in the raw log rather than called a clean run.

## T1 — CUDA bindings initialize driver symbols independently of our probe
Theory: The cuGetProcAddress_v2 reports arise when initializing the CUDA Python
bindings, independently of scheduler execution.
Predicts: Calling cuInit through the bindings in a process with no probe/kernel
will produce the same API reports.
Falsified by: The control is clean, or reports require launching our scheduler.
Cost to check: One import-only sanitizer run.
Status: UNTESTED.

No kernel-source change is proposed on this theory. A subsequent memcheck run
may disable CUDA API error reporting while retaining device-access checks;
both raw and filtered runs must be reported.

T1 status: FALSIFIED. `memcheck_bindings_control.log` reports CUDA_SUCCESS and
zero errors for cuInit alone. Initializing the bindings alone does not reproduce
these reports; do not attribute them to generic import/init behavior.

## T2 — the DSL capability-query path triggers the API reports
Theory: The host capability query used before compiling our probe triggers the
reports independently of executing the probe.
Predicts: Calling get_compute_capability_major_minor in isolation produces
cuGetProcAddress_v2 reports, with no compiled/launched kernel.
Falsified by: The capability-query control is clean.
Cost to check: One sanitizer run.
Status: UNTESTED.

T2 status: FALSIFIED. `memcheck_capability_control.log` returns (10, 0) with
zero errors. The capability query alone also does not reproduce the reports.
The origin of the 41 API reports remains OPEN; neither initial hypothesis is
an established explanation. No production change is made to address them.

Device-access validation: `exhaustion_memcheck_device.log` runs memcheck with
`--report-api-errors no` (device-access instrumentation remains enabled),
passes all five cases, and reports zero errors. The unfiltered log is retained.
Do not summarize the unfiltered run as zero sanitizer errors.
