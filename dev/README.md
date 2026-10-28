# Developer tooling

Everything under `dev/` is tooling for people working on the ports: parity
harnesses against each vendor's stock code, benchmarks, a module tracer, weight
audits and publishing. None of it is part of the served package, none of it is
imported by `src/`, and none of it runs in CI. Most of it drives other projects'
code and checkpoints, so it runs only on a machine that has them.

## Contents

| path | purpose |
|---|---|
| [`oracles/`](oracles/README.md) | parity harnesses: each port against its stock vendor (`vendor_native_parity.py`), AlphaFold 3 against DeepMind's repository (`af3_native_parity.py`), the module-level gates driven by `run_all_parity.sh`, and `fold_check.py` |
| [`bench/`](bench/README.md) | runtime, memory and gradient benchmarks, ours and native |
| [`live/`](live/README.md) | the live-fold display helpers and their GPU smoke tests |
| [`module_trace/`](module_trace/README.md) | records every haiku module's output for one run and diffs two runs |
| [`af3_dev_paths.py`](af3_dev_paths.py), [`af3_dev_paths.sh`](af3_dev_paths.sh) | where the tools find vendor checkouts, venvs and weights (see below) |
| [`audit_coverage.py`](audit_coverage.py) | which tensors of a native checkpoint the converter never reads (level L0 of `run_all_parity.sh`) |
| [`audit_published.py`](audit_published.py) | checks every converted blob before upload; `--apply` loads each into its graph |
| [`publish.py`](publish.py) | uploads converted blobs to the Hugging Face repo the runtime fetches from |
| [`int8_roundtrip_check.py`](int8_roundtrip_check.py) | does an int8 blob dequantise back to its float32 parameters |
| [`openfold3_test.py`](openfold3_test.py) | unit tests for the OpenFold3 to AlphaFold 3 residue-alphabet remap in `converters/openfold3.py` |
| [`AF2_MERGE_NOTES.md`](AF2_MERGE_NOTES.md) | historical: how AF2's monomer and multimer paths were merged |
| [`AF3_DROPOUT_SPEC.md`](AF3_DROPOUT_SPEC.md) | the spec behind the design-time pair-stack dropout |

## Usage

Run from the repository root with this tree's code on the path:

```bash
PYTHONPATH=src:. ~/venv/bin/python dev/audit_coverage.py protenix2 [--checkpoint PATH] [--verbose]
PYTHONPATH=src:.:dev/oracles ~/venv/bin/python dev/audit_published.py --apply   # before every publish
PYTHONPATH=src:. ~/venv/bin/python dev/int8_roundtrip_check.py protenix2 [INT8_DIR]
PYTHONPATH=src:. python dev/publish.py --dir ~/ported --dry_run            # needs huggingface_hub
PYTHONPATH=src:. ~/venv/bin/python -m pytest -q dev/openfold3_test.py      # CPU, seconds
```

`publish.py` takes `--repo`, `--model` and `--precision fp32|fp16|int8`
(both repeatable); it refuses AlphaFold 3's own parameters. Each script's
docstring has the full story.

## Machine paths

Scripts name vendor trees, venvs and weights in a home-relative layout
(`~/venv_boltz/bin/python`, `~/protenix`, `~/ported/<model>`) and resolve them at
run time: Python through `dev_path()` in [`af3_dev_paths.py`](af3_dev_paths.py),
shell through [`af3_dev_paths.sh`](af3_dev_paths.sh). A machine with a different
layout sets environment variables instead of editing scripts:

| variable | meaning | default |
|---|---|---|
| `AF3_DEV_HOME` | root of the layout | `$HOME` |
| `AF3_PORTED_DIR` | converted blobs, `<model>/` under it | `$AF3_DEV_HOME/ported` |

`~/alphafold3` always resolves to the checkout the script lives in, so a worktree
runs its own code. Do not write absolute machine paths into scripts or docs.

## Caveats

* Anything that runs a model wants the GPU, one job at a time: two concurrent
  folds OOM a 23 GB card and inflate each other's timings.
* `~/venv` is the JAX environment. Vendor code runs in its own venv (listed in
  `VENDORS` in [`oracles/vendor_native_parity.py`](oracles/vendor_native_parity.py));
  never install vendor packages into `~/venv`.
