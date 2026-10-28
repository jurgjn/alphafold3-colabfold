# Runtime and memory benchmarks

Runtime and peak-memory measurements of the ported models on this repository's
shared JAX graph, against each vendor's native PyTorch implementation where one
was timed. This folder holds the plots, the TSVs behind them and the scripts that
draw them. It is for anyone choosing a model for a sequence length or a GPU, or
checking a speed claim about a port.

The numbers are from 2026-09-04 and cover nine models: `alphafold3`,
`openfold3`, `openbind0`, `intellifold2`, `opendde`, `boltz2`, `protenix2`,
`rosettafold3` and `chai1`. Models added since then (`protenix1`, `mol1`,
`helixfold3`, the `esmfold2` family, `bioemu2`, `af2_ptm`, `af2_multimer`) have
not been measured here. The TSVs still name `openbind0` by its old name,
`openbind`.

## Contents

| file | what it is |
|---|---|
| `make_plots.py` | draws the runtime plots from the sweep TSVs |
| `plot_memory.py` | draws the memory plots from `mem_a10.tsv` and `sweep_a10.tsv` |
| `plot_protenix2.py` | draws `protenix2_vs_native.png` (dev environment only: it imports `dev/af3_dev_paths.py`) |
| `sweep_a10.tsv` | ours, A10: one row per model x length (column 6: steady seconds) |
| `sweep_a100.tsv` | ours, A100: older and without `openbind0` (see below) |
| `mem_a10.tsv` | ours, A10: peak device memory (column 8, GiB); its timings are not usable |
| `native_sweep.tsv` | native PyTorch, A10: boltz2, rosettafold3, protenix2, opendde, openfold3, chai1 across length |
| `nb_final.tsv` | native boltz-2, 64-384 tokens (`native_boltz2.tsv` is an identical copy) |
| `nr_final.tsv` | native rosettafold3, total time (`native_rf3_total.tsv` is an identical copy) |
| `nr_split.tsv` | native rosettafold3 with featurisation timed separately (`native_rf3_split.tsv` is an identical copy) |
| `runtime_all_A10.png`, `runtime_all_A100.png` | every measured model, linear and log-log |
| `runtime_per_model.png` | one panel per model, per GPU, native where available |
| `ours_vs_native_64.png` | ours against native at 64 tokens, and the ratio |
| `boltz2_vs_native.png`, `rf3_vs_native.png`, `protenix2_vs_native.png` | one model against its native across length |
| `memory_all_A10.png`, `memory_vs_runtime.png` | peak memory against length, and against runtime |

## Regenerating the plots

The scripts need `matplotlib`. Each reads its TSVs from the directory you pass
and writes its PNGs next to itself, i.e. into this folder:

```bash
python docs/benchmarks/make_plots.py docs/benchmarks
python docs/benchmarks/plot_memory.py docs/benchmarks
```

The sweeps themselves are produced by the harnesses under `dev/bench/native/`
(`run_ours_sweep.sh`, `run_mem_sweep.sh`, `bench_sweep.py` and the native
`bench_*.py` runners). They need a GPU to themselves and the vendor
environments described in
[`dev/bench/native/NATIVE_SETUP.md`](../../dev/bench/native/NATIVE_SETUP.md).
None of this runs in CI.

## What is measured

Steady-state seconds per prediction: 3 recycles, 200 diffusion steps, 1 sample,
`num_msa=1024`, single sequence, no template, confidence head on. "Steady state"
is the median of the LAST THIRD of the calls in one process -- see the warm-up
trap below for why not "everything after the first call".

Sequence length is swept with a random sequence; only the shape matters for
timing, and the native harnesses are fed the SAME sequences (same seed, same
alphabet) so both sides fold the same input.

## Provenance of these numbers

The A10 sweep (`sweep_a10.tsv`) was re-measured on 2026-09-04, after that day's
parameter fixes, and includes `openbind0`. Every model landed within +/-0.025 s
of its previous 64-token figure, so those fixes cost no measurable speed.

**The A100 curves are OLDER.** `sweep_a100.tsv` predates that re-measurement and
has no `openbind0` row, so `runtime_all_A100.png` shows eight models where the
A10 plot shows nine. It is a different day's data on a different machine; do not
read an A10-vs-A100 difference off these two plots as if it were controlled.

`opendde` has no 512 point on the A10: it OOMs, asking for 19.53 GiB on a 23 GB
card. That is a result, not a gap -- it is the widest trunk here (pair_channel
384) and runs its diffusion on an expanded structural-token set. The previous
sweep recorded the same failure at the same length.

Three models have no native torch ratio. `alphafold3` has no torch native at all
-- it IS the JAX one, so that comparison is port fidelity, not speed.
`intellifold2`'s two harnesses (torch and jax) are written but have never been
run. `openbind0` needs openfold3>=0.5, which the native OpenFold3 install used
for these runs predates.

## Results

`runtime_all_A10.png` -- all nine models, linear and log-log, 64 to 1024 tokens.

`memory_all_A10.png` / `memory_vs_runtime.png` -- PEAK DEVICE MEMORY, from
`memory_stats()['peak_bytes_in_use']` (the allocator's high-water mark; nvidia-smi
reports what JAX pre-allocated, which is a constant and says nothing). Produced by
`plot_memory.py` from `mem_a10.tsv`, a separate one-call-per-cell pass -- peak is
reached on the first call, so the TIMINGS IN THAT FILE ARE NOT USABLE, only
column 8.

Peak GiB, A10:

```text
model               64     128     192     256     384     512     768    1024
opendde           2.82    3.78    5.31    7.43   13.57     OOM     OOM     OOM
intellifold2      2.58    3.03    3.49    3.91    5.06    7.36   13.39     OOM
boltz2            2.10    2.26    2.68    3.27    4.71    6.88   13.06     OOM
protenix2         2.01    2.24    2.47    2.70    3.66    4.86    8.69   14.04
chai1             1.39    1.53    1.83    2.04    3.00    4.25    7.93   13.04
openfold3         1.55    1.64    1.76    1.88    2.35    3.01    4.82    7.20
rosettafold3      1.56    1.64    1.76    1.88    2.34    2.97    4.83    7.18
openbind0         1.56    1.64    1.76    1.88    2.35    3.01    4.82    7.19
alphafold3        1.23    1.31    1.43    1.55    2.02    2.68    4.50    6.89
```

**Runtime does not predict footprint**, which is the reason to measure both.
boltz2 is 1.5x FASTER than protenix2 at 512 tokens (41.2 s against 60.5 s) and
uses 42% MORE memory (6.88 GiB against 4.86) -- so boltz2 OOMs at 1024 while
protenix2, the slower model, keeps running at 14.04 GiB. A runtime plot cannot
tell you which model will fit on your card.

And it is not a near miss that boltz2 happened to lose: its footprint SCALES
faster. boltz2 goes 6.88 -> 13.06 GiB from 512 to 768 (1.90x for a 1.5x length),
where protenix2 goes 4.86 -> 8.69 -> 14.04 (1.79x then 1.62x). Extrapolating
boltz2 to 1024 gives roughly 20 GiB against the ~22 usable here, so the OOM is
the exponent, not luck.

openfold3, openbind0 and rosettafold3 agree to ~0.02 GiB at every length, which
is what three pair_channel=128 graphs with the same block counts should do; they
are drawn with different dashes because otherwise two of the three vanish under
the third.

`runtime_per_model.png` -- one panel per model, per GPU, native where available.
`boltz2_vs_native.png` -- ours vs native boltz-2, and the ratio by size.

Two things the numbers say that are easy to get wrong:

* **Scaling is NOT linear.** Below ~256 tokens the local exponent is ~1.0; from
  256 to 512 it is ~1.8 (alphafold3: 9.21 -> 18.31 -> 30.97 s at 256/384/512).
  A curve fitted below 384 will badly underestimate long-sequence cost. An
  earlier version of this file claimed "five ports scale linearly" -- that was
  an artefact of stopping at 384.
* **The advantage over native torch shrinks with size, then flattens.** boltz-2:
  2.05x at 64 tokens, 1.53x at 128, ~1.45x at 192-256, 1.59x at 384. It is
  mostly PyTorch eager dispatch overhead, which amortises as arithmetic grows.
  Do not quote the small-size number on its own.

### Native coverage

What `native_sweep.tsv`, the per-model native files and the 64-token figures in
`make_plots.py` hold, all on the A10:

| model | native timing | note |
|---|---|---|
| boltz2 | 64-768, forward vs forward | the clean comparison |
| rosettafold3 | 64-512, featurisation split out at 64/192/384; OOM at 768 | see below |
| protenix2 | 64-512 | `protenix2_vs_native.png`; no timing at 768 |
| openfold3 | 64-768 | preview-2 checkpoint only, see above |
| opendde | 64-256 | no timings recorded from 384 up |
| chai1 | 256 (bucket), 384-768 | chai pads 64 tokens to its 256 bucket, so 64 is not compared |
| intellifold2 | none | harnesses written, never run |
| alphafold3 | none | native IS jax/haiku -- same graph, so a comparison measures port fidelity, not a framework advantage |

`rosettafold3` needed care: its `run()` bundles atomworks featurisation, cif
parsing and writing outputs, while our number is forward-only, so the raw ratio
is an UPPER BOUND. Timing the pipeline separately (`nr_split.tsv`) splits it:

```text
tokens  ours   native total  featurise  native fwd  ratio_total  ratio_fwd
  64    3.12      14.38        2.66       11.72        4.61x      3.76x
 192    7.18      22.19        3.81       18.38        3.09x      2.56x
 384   20.28      61.53        5.34       56.19        3.03x      2.77x
```

So rosettafold3 is ~2.6-2.8x rather than the ~3.0-4.6x the unsplit number
suggests. Incidentally native's featuriser costs 2.7-5.3 s against our 0.9 s.

## Long lengths

The 512 row (A10) puts every model's local exponent in 1.76-2.09, including the
three that read ~1.0 below 256 -- so the exponent is a property of the ARCHITECTURE
at that size, not of the port. What differs between models is the constant factor.
`opendde` cannot run 512 on a 23 GB card at all (`RESOURCE_EXHAUSTED: Out of
memory while trying to allocate 19.53GiB`), which fits its expanded
structural-token diffusion needing more memory per token than the rest.

The spread also widens sharply with length: at 64 tokens the slowest model is
1.4x the fastest; at 512 intellifold2 is 3.0x alphafold3 and opendde is out of
memory. Model choice matters far more on long inputs than short ones.

## Memory ceilings

Read off the peak-memory table above, with ~22 GiB usable on the A10: every
measured model fits 384 tokens and opendde stops there; boltz2 and intellifold2
reach 768 but not 1024; the other six reach 1024. Anything longer, or opendde
beyond 384, needs a larger card.

## Traps, all of which produced wrong numbers here first

* **Concurrent GPU processes.** Four orphaned benchmark processes (~6 GB each)
  made native boltz-2 read 25 s at 128 tokens where the truth is 8.67 s -- and
  the contaminated numbers looked *better* for us, which is how they survived a
  sanity check. Background loops here outlive the turn that starts them and
  stack up. Verify the GPU is EMPTY before every measurement:

  ```bash
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
  ```

* **Warm-up is not always one call.** On an A100 the model retraced on call 2
  (tokamax's lazily-created JAX user context -- see the fix in
  `platform.preinit_tokamax_context`, which `ModelRunner._preinit_tokamax_context`
  points at), so call 2 cost ~45 s. Taking "the median after the first call"
  reported 27.8 s where steady state was 1.9 s. Always print every call, not
  just the summary.

* **Both caches or neither.** `run_alphafold.py` defaults
  `--cache_dir=/tmp/alphafold_cache` and wires the JAX compilation cache and the
  tokamax autotune cache. A harness that builds the model itself gets neither
  and pays a full cold compile every process (56.3 s vs 8.1 s warm, from a
  3.9 MB cache). A one-pass SWEEP cannot benefit either way -- each length is a
  new executable -- but anything repeating a shape can.

* **`Autotuning cache miss` costs nothing.** tokamax logs it and falls back to a
  heuristics config (`ops/op.py:514-523` in tokamax). Do not "fix" it by calling
  `tokamax.autotune()` per shape; that adds real search time.

* **MSA depth is not a runtime axis.** Featurisation pads `msa` to 16384 rows
  whatever the input carries and the model truncates to `config.evoformer.num_msa`,
  so input depth changes a scalar (`num_alignments`) and no tensor shape. The
  config knob does cost time -- 10% of the trunk at 64 tokens, 3.9% at 256 --
  but under 1% of a full fold, because ~93% of that is diffusion sampling.

* **Feed native the same input you feed us.** Handing native rf3 `6MRR.pdb`
  makes atomworks split the waters into a second chain, so it folds more than
  our 68 tokens. Strip to protein-only first.

* **Not every comparison is like-for-like.** Native boltz-2 is timed on a
  pre-built batch, so it is forward-vs-forward with ours. Native rosettafold3's
  `run()` includes atomworks featurisation, cif parsing and writing outputs, so
  its number is an UPPER BOUND on the ratio, not a measurement of it. The A100
  series also runs tokamax Triton kernels an A10 cannot, so A10-vs-A100 differs
  in kernel path as well as hardware.
