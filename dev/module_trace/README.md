# Module tracer

Run one input through one model and record **every haiku module's output**, so two
runs can be diffed module by module. Built to answer one question fast: *which
module is the first to differ?* It is a debugging tool for people changing the
graph or a converter; nothing in the served package imports it.

## Usage

Run from the repository root, as a module of `dev/`, with this tree's `src/` on
the path:

```bash
PYTHONPATH=src:. ~/venv/bin/python -m dev.module_trace list
PYTHONPATH=src:. ~/venv/bin/python -m dev.module_trace run --model openfold3 --case 6mrr_single_seq --label baseline
PYTHONPATH=src:. ~/venv/bin/python -m dev.module_trace compare ~/module_traces/openfold3/6mrr_single_seq/{baseline,after_change}
PYTHONPATH=src:. ~/venv/bin/python -m dev.module_trace show ~/module_traces/openfold3/6mrr_single_seq/baseline
PYTHONPATH=src:. ~/venv/bin/python -m dev.module_trace table
```

`list` shows every case and model and whether its inputs exist on this machine.
`table` prints one row per stored trace. Traces go to
`~/module_traces/<model>/<case>/<label>/` (`--out` overrides; `~` resolves through
[`dev_path`](../af3_dev_paths.py), so `AF3_DEV_HOME` moves it). `run` takes
`--num-recycles` (default 0), `--diffusion-steps` (200) and `--seed` (1);
`compare` takes `--tol` (1e-6), `--section modules|outputs` and `--limit`.

`run` executes the model, so it wants a GPU and the same serialisation as the
[parity oracles](../oracles/README.md): one job on the card at a time.

**Known breakage (2026-10-28):** `run` currently stops at `models.build` with
`ModuleNotFoundError: No module named 'colabdesign2'`. `_repo_table` in
[`models.py`](models.py) still imports the ColabDesign2 package to find its
checked-in shape tables, and that package does not exist in this repository.
`list`, `show`, `compare` and `table` are unaffected.

## Contents

| file | purpose |
| --- | --- |
| [`__main__.py`](__main__.py) | the CLI: `list`, `run`, `compare`, `show`, `table` |
| [`trace.py`](trace.py) | builds the case and model, intercepts every module call, writes the trace |
| [`compare.py`](compare.py) | walks two traces in execution order and reports the first divergence |
| [`cases.py`](cases.py) | the input registry: each case's batch, reference structure and declared channels |
| [`models.py`](models.py) | the traceable models and their checkpoints, built by init plus merge so a partial port still traces |
| [`gen_param_shapes.py`](gen_param_shapes.py) | generator for parameter-shape tables; its usage line and output directory still name ColabDesign2's `tools/` and `colabdesign2/` layout, which this repository does not have |

## How it captures

`hk.intercept_methods` wraps every module call, so the trace *is* the module tree
-- nothing is instrumented by hand and nothing goes stale when the graph changes.
The trunk recycles run in a `fori_loop` and the pairformer/diffusion stacks in
scans, so at trace time every activation is a tracer; each output leaf is
therefore handed to a `jax.debug.callback`, which fires with concrete values at
run time, inside loops and scans included. A module that runs 48 times (one per
pairformer block) records 48 entries under the same key, in execution order.

Each array is fingerprinted inside the callback and dropped: shape, dtype,
mean/std/min/max/absmax, fraction of zeros, count of non-finite values, and a
fixed-seed random projection of the flattened array. The projection is what makes
"identical" mean identical -- a transpose, a permutation or one changed element
moves it, where mean/std easily agree. Tracing a 48-block trunk costs kilobytes.

`--max-calls-per-site N` (default 8) bounds the cost of repetitive loops: all
calls are **counted**, the first N are fingerprinted. `--save-arrays` additionally
writes `modules.npz` with the arrays themselves (<= 8 MB each) for hands-on
debugging.

## What a trace holds

`manifest.json`:

| field | meaning |
| --- | --- |
| `modules` | call site -> `{calls, fp[]}`, in execution order (`call_order`) |
| `outputs` | the model's own returned leaves, fingerprinted the same way |
| `inputs` | per batch key: shape, dtype, fraction non-zero, distinct values |
| `channels_live` / `channels_unmet` | which optional inputs (msa, template, ligand, multimer) are actually populated, vs what the case declared it exercises -- this is how "the extra input silently did nothing" gets caught |
| `metrics` | CA-RMSD to the case's native structure, CA-CA spacing, pLDDT/PAE |
| `params_left_at_init` | weights the converter did not supply (unported heads) |

## Comparing

`compare` walks A's execution order and reports `MATCH` / `CLOSE` / `DIFFER` /
`ONLY_A` / `ONLY_B` per call site, plus **`FIRST DIVERGENCE`** -- the earliest
differing module, which is the bug site; everything after it is downstream
contamination. A changed call count is itself a difference (the graph changed).

## Cases and models

Cases live in `cases.py`, and every registered case is active. The workhorse is
`6mrr_single_seq` (68-residue protein, single sequence, no template or MSA), with
`6mrr_self_template` beside it. The rest are deliberately taken from AlphaFold3's
own canonical inputs rather than invented: `af3_featurised_example` (DeepMind's
featurised test batch -- real 16384-row MSA, 4 templates, polymer-ligand bonds)
and the 13 `alphafold3/examples` JSONs (`af3_*`) covering homodimer,
protein-protein, CCD ligand + covalent bond, SMILES ligand, ions, DNA duplex,
methylated DNA, RNA hairpin, modified RNA, glycosylation and phosphorylation.
A few structures with a known answer follow (`1stp_streptavidin`, `1ehz_trna`,
`7bny_emcv_2a`, `lig_btn`, `lig_atp`, `lig_hem`). `list` prints the current set.

Models come from `models.py` (`CHECKPOINTS`: alphafold3, openfold3, intellifold2,
opendde, boltz2, protenix2, rosettafold3, chai1), loaded by the init+merge path so a partially ported model still traces, with the unported
scopes listed in `params_left_at_init`.
