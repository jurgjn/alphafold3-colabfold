# Benchmarks

Runtime, memory, gradient and fold-quality measurements of the ports, and the
native-vs-ours timing comparisons. For anyone producing or checking a
performance number. Nothing here is a pass/fail gate; parity lives in
[`../oracles/`](../oracles/README.md).

## Contents

| path | purpose |
|---|---|
| [`native/`](native/NATIVE_SETUP.md) | ours-vs-native timing: per-vendor harnesses, length and memory sweeps, recorded TSVs. Start with [`NATIVE_SETUP.md`](native/NATIVE_SETUP.md), which holds the install recipes and every trap that produced a wrong number |
| `results/` | Colab and GPU gate notebooks with their executed `*_output.ipynb`, the 2026-09-17 sweep TSVs and plot (`plot_vs_native.py`, `ours_vs_native.png`), and the small probes they ran (`foldtest.py`, `jaxpin.py`, `slim_check.py`, `builddata.py`, `probe.py`); `make_e2e.py` builds `e2e.ipynb` |
| [`sweep22.sh`](sweep22.sh) | `fold_check.py` on 6MRR for every model, one line each; `sweep22_2026-09-06.txt` is a recorded run |
| [`grad_sweep.sh`](grad_sweep.sh), [`grad_esm.sh`](grad_esm.sh) | finite-difference gradient checks (`dev/oracles/grad_check.py`) per model, one process each; the `*_2026-09-07.txt` files are recorded runs |
| [`glu_grad.py`](glu_grad.py) | which GLU a design run should take (tokamax, Pallas, none), forward and backward; the answer is in its docstring |
| [`e2e_esm.py`](e2e_esm.py) | soft sequence through ESM-C, the trunk and the distogram with one gradient; peak memory with both weight sets resident |
| [`esm_design.py`](esm_design.py) | does ESMFold2 design descend, with and without the language model in the gradient chain |
| [`blob_ab.py`](blob_ab.py) | fold 6MRR from two weight directories and compare CA-RMSD and peak memory |
| [`l2_native_probe_protenix.py`](l2_native_probe_protenix.py) | can protenix's token diffusion transformer be built from the checkpoint and run on synthetic inputs |

## Usage

From the repository root. The shell drivers source
[`../af3_dev_paths.sh`](../af3_dev_paths.sh) and take an output file:

```bash
bash dev/bench/sweep22.sh                        # writes dev/bench/sweep20.txt
bash dev/bench/grad_sweep.sh grad.txt            # per-model logs in dev/bench/out/
PYTHONPATH=src ~/venv/bin/python dev/bench/glu_grad.py 256 512
```

The Python scripts carry their own command lines in their docstrings.

## Caveats

* GPU only, one measurement at a time, on a quiet host: concurrent jobs and host
  CPU load both inflate timings without failing (see
  [`native/NATIVE_SETUP.md`](native/NATIVE_SETUP.md#traps-that-produced-wrong-numbers)).
* Native harnesses run in the benchmark venvs described in `NATIVE_SETUP.md`,
  not in the parity venvs.
* `results/run_native_sweep.sh` looks for the `bench_*.py` harnesses beside
  itself, but they live in `native/`; run them from there until it is fixed.
* Recorded `.txt` / `.tsv` files are dated measurements, not current numbers.
