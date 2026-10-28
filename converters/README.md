# Weight converters

Offline tooling that turns a published AF3-family checkpoint into the haiku
parameter blob (`<model>.bin.zst`) that `run_alphafold.py` loads. It is for
whoever adds a model or re-converts one; a normal run never needs it.

**This package is deliberately outside `src/alphafold3`.** The served model path
never imports torch, never downloads a 2 GB checkpoint, and never runs a
converter: it loads already-converted weights (from a local `--model_dir`, or
pulled from Hugging Face). Conversion happens once, here, and the product is
published.

## Usage

```bash
# fetches the original checkpoint into --out, converts, writes <model>.bin.zst
PYTHONPATH=src ~/venv/bin/python -m converters.convert --model boltz2 --out ~/ported/boltz2

# or convert a checkpoint you already have
PYTHONPATH=src ~/venv/bin/python -m converters.convert --model openfold3 \
    --checkpoint ./of3-p2-155k.pt --out ~/ported/openfold3
```

`--model` takes any name in the registry below (`CONVERTERS` in
[`__init__.py`](__init__.py)); `--help` lists them.

**Requirements.** Conversion reads PyTorch checkpoints, so it needs `torch` (a
CPU build is enough; `~/venv` has one), and the blob writer imports
`alphafold3`, so `src` must be on `PYTHONPATH`. No GPU. None of this runs in CI.

## Models

Every name in the registry, with the checkpoint `converters.convert` fetches
when `--checkpoint` is omitted. The sources are in [`sources.py`](sources.py),
which also records why each one was chosen.

| `--model` | module | model | checkpoint source |
|---|---|---|---|
| `openfold3` | `openfold3.py` | OpenFold3 preview-2 (AlQuraishi Lab) | `openfold.s3.amazonaws.com/staging/of3-p2-155k.pt` |
| `openbind0` | `openfold3.py` | OpenBind, OpenFold3 v0.5.0 | `openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt` |
| `intellifold2` | `intellifold2.py` | IntelliFold-v2 (IntelligenAI) | HF `intelligenAI/intellifold`, `intellifold_v2.pt` |
| `opendde` | `opendde.py` | OpenDDE (Aureka Research) | HF `aurekaresearch/OpenDDE` at commit `eddd563c`, `opendde.pt` |
| `boltz2` | `boltz2.py` | Boltz-2 (jwohlwend/boltz) | HF `boltz-community/boltz-2`, `boltz2_conf.ckpt` |
| `protenix2` | `protenix2.py` | Protenix-v2 (ByteDance) | HF mirror `TMF001/protenix-v2-weights`, `protenix-v2.pt` |
| `protenix1` | `protenix2.py` | Protenix-v1 (ByteDance) | `protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt` |
| `mol1` | `mol1.py` | Mol-1 (Molence Bio), a Protenix-v2 fork | `mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt` (the URL `mol pred` downloads) |
| `rosettafold3` | `rosettafold3.py` | RoseTTAFold3 (RosettaCommons) | `files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt` |
| `chai1` | `chai1.py` | chai-1 (Chai Discovery) | `chaiassets.com` (5 TorchScript archives), plus `distogram_head.pt` from `sokrypton/chai-lab@dgram` |
| `helixfold3` | `helixfold3.py` | HelixFold3 (Baidu PaddleHelix) | `paddlehelix.bd.bcebos.com/HelixFold3/params/HelixFold3-params-20250714.zip` (non-commercial terms) |
| `bioemu2` | `bioemu2.py` | BioEmu2 preview (Microsoft Research) | HF `microsoft/bioemu2`, `checkpoints/bioemu2-preview/checkpoint.ckpt` |
| `esmfold2` | `esmfold2.py` | ESMFold2 (Arc Institute / CZ Biohub) | HF repo `biohub/ESMFold2` (whole snapshot) |
| `esmfold2_fast` | `esmfold2.py` | ESMFold2-Fast | HF repo `biohub/ESMFold2-Fast`; as above |
| `esmfold2_lm600m` | `esmfold2.py` | ESMFold2 on ESM-C 600M | HF repo `Synthyra/ESMFold2-600`; as above |
| `esmfold2_lm300m` | `esmfold2.py` | ESMFold2 on ESM-C 300M | HF repo `Synthyra/ESMFold2-300`; as above |
| `esmc` | `esm_lm.py` | ESM-C 6B, the tower ESMFold2 reads | HF repo `biohub/ESMC-6B`; as above |
| `esm2` | `esm_lm.py` | ESM2 3B, the tower chai-1 reads | not downloaded alone: it ships with chai-1's assets as `traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`; pass it as `--checkpoint` |

**The ESM rows download whole Hugging Face snapshots.** `sources.py` records
the ESMFold2 and ESM-C sources as repos (`repo=`), which `convert.fetch_checkpoint`
fetches with `huggingface_hub.snapshot_download` into `--out`; `esm2` (`local=`)
is not downloadable alone and needs `--checkpoint`. When passing `--checkpoint`
for the ESMFold2 family, give the snapshot directory, not the `.safetensors`
file: the converter checks the registry row against the release's
`config.json`, and writes the language-model shim as `<model>.lm.npz` beside the
blob. `sources.py` also lists the `esmc_300m` and `esmc_600m` towers
(`biohub/ESMC-300M-1500000`, `biohub/ESMC-600M-1500000`), which have no registry
entry of their own; convert them with
`esm_lm.convert_esmc_weights(ckpt, out, tower='esmc_300m')`.

AlphaFold3's own weights are not converted: DeepMind publishes a haiku blob
already, and its terms require you to request it.

## Contents

| file | purpose |
|---|---|
| `convert.py` | the CLI above: fetch, convert, report the blobs written |
| `__init__.py` | `CONVERTERS`, the registry from model name to entry point, imported lazily |
| `sources.py` | where each original checkpoint comes from, and why |
| `common.py` | primitives every family shares, and the blob writer and reader |
| `<model>.py` | one converter per model or family, as in the table above |
| `esm_lm.py` | weight maps for the two protein-language-model towers |
| `quantise.py` | rewrite a blob as fp16, bf16 or int8 |
| `pdb.py` | minimal CA reader the converters use for checks |
| `esm_lm_layout_test.py` | the two ESM-C checkpoint layouts convert to the same tower |

`common.py` holds the primitives every family shares -- the transpose/reshape
math is the same everywhere, only the torch leaf names and the fusion mode
differ, so each family is a `Dialect` plus a map.

### The language-model towers

`esm_lm.py` maps two whole models rather than one family's blocks. Two of the
ports fold from a protein language model -- chai-1 from ESM2 3B, ESMFold2 from
ESM-C -- and neither runs at full fidelity without one, so the towers are
converted and run in jax like everything else. Only the WEIGHT MAPS are here:
running a tower is inference, so the forward, the vocabularies and the loader
live in `alphafold3/model/esm.py`, for the same reason
`params._dequantise_records` does not import this package. The two are one file
because they are one architecture: pre-LayerNorm, rotary, head_dim 64,
differing in five documented places (fused vs split qkv, q/k norm, SwiGLU vs
gelu, residual scaling, and how many hidden states the consumer reads). The
ESMFold2 shim that turns ESM-C's states into a pair representation is not part
of the tower and lives in `esmfold2.py`.

Both towers convert at int8 (`esm_lm.convert_tower` quantises in-line) and run
under a `lax.scan`. For ESM-C that is not an optimisation: float32 cannot be
written at all (a record header packs its length as a signed 32-bit int, and the
fused fc1 stack is one 5.27 GiB tensor), and a python loop over the 80 blocks
leaves the whole 25 GB tower resident on a 23 GB card.

## Smaller blobs: `quantise.py`

A converted blob is float32 while the model runs in bfloat16. `quantise.py`
rewrites an existing blob under a smaller storage scheme:

```bash
PYTHONPATH=src ~/venv/bin/python -m converters.quantise \
    ~/ported/boltz2/boltz2.bin.zst ~/ported/boltz2/boltz2.int8.bin.zst int8
```

| scheme | storage |
|---|---|
| `fp32` | unchanged |
| `fp16` | float16; an array outside float16's range stays float32 |
| `bf16` | bfloat16, stored as its uint16 bit pattern |
| `int8` | int8 with per-channel, per-row-block scales for float matrices of at least 65536 elements; float16 for the other float arrays |
| `int8-bf16` | the same with bfloat16 for the rest, kept to measure the two |

Integer arrays are never touched. The loader dequantises through
`alphafold3.model.params`, so nothing downstream knows the difference. The cost
of int8 is measured, not assumed -- see
[`docs/ported_models.md`](../docs/ported_models.md). Name the output
`<model>.<precision>.bin.zst`; that is the file `dev/publish.py --precision`
looks for.

## Coverage audit

The direction haiku already guards -- a parameter the graph wants that the blob
lacks -- fails at load. That is blind to the opposite leak: **a trained tensor
in the checkpoint that the graph has no slot for**. Stock AF3 creates its norms
with `create_offset=False` and its Linears with `use_bias=False`, so wherever a
ported family added an offset or a bias, the converter simply never asks for the
key and nothing anywhere fires. `dev/audit_coverage.py` checks that direction:

```bash
PYTHONPATH=src:. ~/venv/bin/python dev/audit_coverage.py protenix2 [--checkpoint PATH] [--verbose]
```

It knows the protenix1/2, mol1, openfold3, openbind0, rosettafold3,
intellifold2, boltz2, opendde, chai1 and ESMFold2 checkpoints (`_LOADERS`, with
default paths through `dev_path`); bioemu2, helixfold3 and the ESM towers are
not in it.

It wraps the state dict to record every key addressed by name, then resolves the
rest by VALUE against the converted tree -- two converters scan with `sd.items()`
instead of indexing, and a scan cannot be attributed key by key. The value index
knows the three fusions this codebase uses (SwiGLU `concat_ab` halves, triangle
`interleave_ab` even/odd columns, and `layer_stack`/`stack_super` slices to two
levels), because a source tensor hidden inside a fused leaf is not missing.

**Every hit still needs triage** -- the tool proves a tensor does not appear
verbatim, not that it matters. Four outcomes recur:

* *duplicate* -- openfold3's checkpoint carries the whole diffusion module twice,
  under `diffusion_module.*` and `sample_diffusion.diffusion_module.*`,
  bit-identical. The converter reads one. Nothing is lost.
* *inert* -- an offset on a LayerNorm whose output feeds only an attention-logit
  projection adds the same constant to every key in a head, and softmax is
  shift-invariant. rosettafold3's 30 `ln_0.bias` and boltz2's 30 `*_proj_z.N.0.bias`
  are all of this kind. Leave them, and say so, or someone re-derives it.
* *sliced* -- a fused source projection the converter cuts into pieces by COLUMN.
  opendde's `linear_no_bias_f` (128, 385) becomes ref_mask(1) + ref_element(128) +
  atom-name-chars(256), and an arbitrary column range is not something the value
  index can enumerate, so both instances report as missing. This is the tool's
  known false positive: check the converter before believing a fused-projection
  hit.
* *real* -- see the table below.

### What the audit found (2026-09-04) -- all `real` rows now fixed

| model | tensor | verdict |
|---|---|---|
| protenix2 | `distogram_head.linear.bias` (64) | **real** |
| rosettafold3 | `distogram_head.predictor.bias` (65) | **real** |
| boltz2 | `distogram_module.distogram.bias` (64) | **real** |
| opendde | `distogram_head.linear.bias` (96) | **real** |
| rosettafold3 | `atom_attention_encoder.process_s_trunk.0.bias` (384) | **real** |
| rosettafold3 | `atom_attention_encoder.process_z.0.bias` (128) | **real** |
| boltz2 | `single_conditioner.single_embed.bias` (768) | **real** |
| rosettafold3 | `to_r_update.0.bias` (128) | cosmetic: a constant added to every atom's position update is a rigid translation |
| rosettafold3 | 30 x `ln_0.bias` | inert (softmax shift) |
| boltz2 | 30 x `*_proj_z.N.0.bias` | inert (softmax shift) |
| protenix2 | `confidence_head.{lower,upper}_bins` | not trained: bin edges, and they match AF3's 39 defaults |
| boltz2 | `bfactor_module.*` | out of scope: no B-factor head |
| openfold3 | 8 under `sample_diffusion.*` + `version_tensor` | duplicate |
| opendde | 2 x `linear_no_bias_f` (128, 385) | sliced (see above) |
| intellifold2 | -- | **0 unmapped** |

The distogram bias is one shared fix: `half_logits` is built by
`distogram_head.py`'s stock path with no bias, so four families lose a trained
one. It is not small -- protenix2's spans -2.05..+1.82, and native adds
`b + b^T`, so `softmax(2b)` alone spreads 0.50 to 2e-4 across bins. Blast radius
is the distogram outputs and `contact_probs` only; diffusion never reads these
logits, which is why no fold gate could catch it.

Re-running the audit after the fixes: protenix2 3 -> 2, rosettafold3 37 -> 33,
boltz2 37 -> 35, opendde 5 -> 4.

**Fix every copy, and measure in the one your gates import.** The package
converters here are what builds a published blob, but at the time every
ColabDesign2 oracle converted in-process through `colabdesign2/af3/converters/`
(its `CHECKPOINTS` point at the raw torch files, so it never reads `~/ported`).
Patching only this directory left every gate running the old conversion and
reporting a pass; the isolation that caught it read `distogram bias absmax
0.000`. The trees had really diverged, so patch each -- do not copy one over the
other.

**Two of the fixes needed the native FORWARD read, not just the tensor name.** Our
graph applies the distogram linear and then symmetrises, passing the bias through
twice. protenix2 and opendde symmetrise after the linear too, so their bias maps
straight across; rosettafold3 and boltz2 symmetrise the pair features FIRST, so
theirs enters once and the converter halves it. Name-matching alone would have got
two of the four wrong by a factor of two.

Run the audit on every new converter. The idea is borrowed: an independent
Protenix port by ChoongHwanLee (`chlee19990109-cloud/ColabFold`, branch
`colabfold2-protenix-proof`) asserts all 4174 of its source tensors are consumed
exactly once, and that discipline is what surfaced this whole table.

**A stronger check we do not have yet:** they also round-trip the state dict
through the *native* model (`load_state_dict` -> 0 missing / 0 unexpected), which
distinguishes a tensor we drop from one the native code ignores too. Without it,
a dead tensor left in a checkpoint by an older training run reads as a gap.

## Tests

`dev/openfold3_test.py` covers the conventions that are silent when wrong -- the
residue-alphabet permutations, the two features_1d classes AF3 lacks, and the
i/j crossing between AF3's two pair-embedding sites (see
[`docs/of3_af3_porting_notes.md`](../docs/of3_af3_porting_notes.md)).
`esm_lm_layout_test.py` checks the two ESM-C layouts. Both run on CPU in
seconds:

```bash
PYTHONPATH=src:. ~/venv/bin/python -m pytest dev/openfold3_test.py converters/esm_lm_layout_test.py -q
```

## Publishing

`run_alphafold.py` fetches a ported model's weights on first use from the
Hugging Face repo `_WEIGHTS_REPO` in `alphafold3.model.model_registry`
(`sokrypton/af3-any-model`), by exact filename over plain HTTPS. The repo is
grouped by family (`WEIGHTS_FOLDERS`; a model not listed there is its own
folder), and the language-model towers share `lm/`:

```text
<repo>/
  openfold3/   openfold3.bin.zst  openbind0.bin.zst
  protenix/    protenix1.bin.zst  protenix2.bin.zst
  esmfold2/    esmfold2.bin.zst   esmfold2.lm.npz   (and the other variants)
  lm/          esmc.bin.zst  esm2.bin.zst  ...
  boltz2/      boltz2.bin.zst
  ...          one folder each for bioemu2, chai1, helixfold3, intellifold2,
               mol1, opendde, rosettafold3
```

`dev/publish.py` uploads a local tree laid out as `<dir>/<model>/<files>`.
`--precision fp16` or `--precision int8` adds the quantised blob beside the
float32 one, and `--dry_run` lists the files without uploading. It needs
`huggingface_hub` (in the system `python3`) and a token (`huggingface-cli login`
or `HF_TOKEN`). Where `alphafold3` is not importable it cannot read the default
repo from the registry, so pass `--repo`:

```bash
python3 dev/publish.py --dir ~/ported --repo sokrypton/af3-any-model --model boltz2 --dry_run
```

AlphaFold 3's own parameters are not published here and never will be: DeepMind
requires you to request them.
