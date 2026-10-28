# Docker

The container recipe from DeepMind's AlphaFold 3, unchanged in this fork. It
builds HMMER 3.4 with the Jackhmmer sequence-limit patch, installs the
locked dependency tree with uv (all dependency groups, so JAX with CUDA 12,
Haiku and RDKit), and builds the chemical components data. The full walkthrough,
from GPU drivers to the first prediction, is in
[docs/installation.md](../docs/installation.md).

## Contents

| File | Purpose |
|---|---|
| [`Dockerfile`](Dockerfile) | The image: CUDA 12.6.3 on Ubuntu 24.04, Python 3.12, HMMER, this package in `/alphafold3_venv`, and the XLA environment variables from [docs/performance.md](../docs/performance.md). |
| [`jackhmmer_seq_limit.patch`](jackhmmer_seq_limit.patch) | Adds `--seq_limit` to HMMER's Jackhmmer, applied before HMMER is built. |
| [`dockerignore`](dockerignore) | Carried from upstream. Docker does not read it under this name (it reads `.dockerignore` at the build-context root, or `docker/Dockerfile.dockerignore`), so the whole checkout is the build context. |

## Building and running

Build from the repository root, which is the build context:

```sh
docker build -t alphafold3 -f docker/Dockerfile .
```

Run any `--model` the same way as AlphaFold 3. AlphaFold 3 itself needs its
parameters mounted and passed with `--model_dir`; the other models download
their converted weights on first use to `/root/.cache/alphafold3/weights` inside
the container, so mount a directory there to keep them between runs:

```sh
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume $HOME/af3_weights:/root/.cache/alphafold3/weights \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --model=boltz2 \
    --json_path=/root/af_input/fold_input.json \
    --output_dir=/root/af_output \
    --use_msa_server \
    --norun_data_pipeline
```

For GPUs of CUDA capability 7.x (V100, T4), override `XLA_FLAGS` as described
in [docs/performance.md](../docs/performance.md#cuda-capability-7x-gpus).
