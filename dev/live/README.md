# Live fold frames

Display helpers and smoke tests for watching a fold as it runs.
`ModelRunner.live_model()` in [`run_alphafold.py`](../../run_alphafold.py) calls back
after every trunk pass and every denoise step; the helpers here turn those
callbacks into things a notebook can draw (contact maps, one mmCIF model per
frame) and into simple steering potentials for `on_step`. For anyone working on
the live or steering path. Nothing here is on the fold path.

## Contents

| file | purpose |
|---|---|
| [`live_frames.py`](live_frames.py) | `as_batch`, `contact_map`, `recycle_distance`, `frame_cif`, `frame_positions`, and the steering helpers `pull_together` and `symmetrise` |
| [`live_frames_test.py`](live_frames_test.py) | unit test: a structural-token model's frames are mapped back to the residue layout. CPU, no weights |
| [`live_test.py`](live_test.py) | drives recycles and diffusion steps from Python on openbind0, printing every frame |
| [`steer_test.py`](steer_test.py) | does `on_step` steer: pulls two distant residues together |
| `live_colab.ipynb` | an older Colab check of the live path |

## Usage

From the repository root:

```bash
~/venv/bin/python -m pytest -q dev/live/live_frames_test.py   # CPU, seconds
~/venv/bin/python dev/live/live_test.py                        # GPU
~/venv/bin/python dev/live/steer_test.py                       # GPU
```

## Caveats

* `live_test.py` and `steer_test.py` need the GPU, converted openbind0 weights at
  `~/ported/openbind0` (resolved through [`dev_path`](../af3_dev_paths.py)), and
  an input file `dyn.json` beside them, which is not checked in: supply any
  AlphaFold 3 fold-input JSON under that name. They write a compilation cache
  (`cache_live/`, `cache_cost/`) into this directory.
* `live_colab.ipynb` overlays files from the retired `colab` branch onto an old
  wheel; treat it as a record, not a runnable test.
* Not in CI.
