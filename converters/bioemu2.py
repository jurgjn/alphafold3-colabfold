"""BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) -> haiku parameters.

A rename, not a remap: network/bioemu2.py reads the vendor's own tensors in the
vendor's own layouts. One scope, `diffuser/~/bioemu2`, whose parameter names
are the EMA state-dict keys with the `module.` prefix dropped and '.' -> '/':

  * `<...>.layers.<i>.<rest>` of the three layer stacks (4 MSA, 48 PairMixer,
    16 dual-stream) is STACKED on a leading axis as `<...>/layers/<rest>`,
    which the network runs under lax.scan;
  * any other numbered component, an nn.Sequential or a short ModuleList,
    becomes `name_<i>` (`cond.2.weight` -> `cond_2/weight`).

Every tensor in the checkpoint is either converted or named in _DROPPED with its
reason, and the conversion writes network/bioemu2_params.json, the shape table
the graph builds itself from, so the two cannot drift apart.
"""

import json
import os
import re

import numpy as np

from converters import common as C

SCOPE = 'diffuser/~/bioemu2'
STACKED = ('sequence_embedder.msa_module.layers', 'sequence_embedder.pairmixer_module.layers',
           'structure_module.layers')
# In the checkpoint, never read by the forward pass of a single-device run.
_DROPPED = {
    'n_averaged': 'AveragedModel bookkeeping, not a weight',
    # BioMoaCModel builds a second token-bond projection that compute_trunk_embeddings
    # never calls (the trunk uses sequence_embedder.token_bonds); only the
    # context-parallel wrapper copies it.
    'token_bonds.weight': 'created by BioMoaCModel, never used by the serial forward',
}
SHAPES_JSON = os.path.join(os.path.dirname(__file__), '..', 'src', 'alphafold3', 'model',
                           'network', 'bioemu2_params.json')


def haiku_name(key):
  """`a.b.layers.3.c.0.weight` -> (`a/b/layers/c_0/weight`, 3); None for unstacked."""
  for stack in STACKED:
    if key.startswith(stack + '.'):
      rest = key[len(stack) + 1:]
      idx, rest = rest.split('.', 1)
      return stack.replace('.', '/') + '/' + _flat(rest), int(idx)
  return _flat(key), None


def _flat(key):
  return re.sub(r'\.(\d+)(?=\.|$)', r'_\1', key).replace('.', '/')


def map_bioemu2(sd):
  """{torch key: array} -> {haiku name: array}."""
  out, stacks = {}, {}
  for key, value in sd.items():
    if key in _DROPPED:
      continue
    name, idx = haiku_name(key)
    if idx is None:
      if name in out:
        raise ValueError(f'two checkpoint keys map to {name}')
      out[name] = np.asarray(value, np.float32)
    else:
      stacks.setdefault(name, {})[idx] = np.asarray(value, np.float32)
  for name, layers in stacks.items():
    n = len(layers)
    if sorted(layers) != list(range(n)):
      raise ValueError(f'{name}: layer indices {sorted(layers)} are not 0..{n - 1}')
    out[name] = np.stack([layers[i] for i in range(n)])
  return out


def load_state_dict(checkpoint):
  import torch
  ck = torch.load(os.path.expanduser(str(checkpoint)), map_location='cpu', weights_only=True)
  sd = ck['model_ema']
  out = {}
  for k, v in sd.items():
    k = k[len('module.'):] if k.startswith('module.') else k
    out[k] = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
  return out


def convert_bioemu2_weights(checkpoint, output_dir, write_shapes=True):
  """bioemu2-preview checkpoint.ckpt -> output_dir/bioemu2.bin.zst."""
  sd = load_state_dict(checkpoint)
  unknown = [k for k in sd if k not in _DROPPED and not _converted(k)]
  if unknown:
    raise ValueError(f'unmapped checkpoint keys: {unknown[:5]}')
  params = map_bioemu2(sd)
  n_in = sum(v.size for k, v in sd.items() if k not in _DROPPED)
  n_out = sum(v.size for v in params.values())
  if n_in != n_out:
    raise ValueError(f'{n_in} checkpoint values in, {n_out} out')
  shapes = {k: list(v.shape) for k, v in sorted(params.items())}
  if write_shapes:
    with open(SHAPES_JSON, 'w') as fh:
      json.dump(shapes, fh, indent=0, sort_keys=True)
  elif os.path.exists(SHAPES_JSON):
    with open(SHAPES_JSON) as fh:
      if json.load(fh) != shapes:
        raise ValueError('this checkpoint does not match network/bioemu2_params.json')
  C.write_params_blob(output_dir, 'bioemu2.bin.zst', {SCOPE: params})
  return output_dir


def _converted(key):
  return True   # every key not dropped maps by rule; collisions raise in map_bioemu2
