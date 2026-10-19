"""The UPSTREAM half of af3_native_parity.py. Runs in ~/venv_af3, never here.

DeepMind's package is also called `alphafold3`, so it cannot share a process
with ours: whichever is first on sys.path wins every import, silently. This
script is therefore run as `~/venv_af3/bin/python -I af3_upstream_runner.py`
-- `-I` keeps both PYTHONPATH and this directory off sys.path, so the only
`alphafold3` it can import is the one installed from ~/af3_upstream.

  featurise IN.json OUT.npz          upstream's own featurisation of a fold input
  forward   IN.npz  OUT.npz          the model on a batch, with injected samples

FORWARD runs the real `Model.__call__` -- trunk, recycles, distogram,
confidence -- so every module is built in the order and scope the shipped model
builds it. That matters: 435b8dc was a haiku name bound by CREATION ORDER, and a
harness that constructs one module alone cannot see that class of bug. The one
override is `_sample_diffusion`, the only stochastic part: it returns the
injected coordinates instead of sampling, and calls the real diffusion module
once on an injected (positions_noisy, noise_level) so one denoise step is
compared on identical inputs.
"""
import sys

import numpy as np


def _flatten(tree, prefix=''):
  out = {}
  if isinstance(tree, dict):
    for k, v in tree.items():
      out.update(_flatten(v, '%s%s/' % (prefix, k)))
  elif tree is not None and hasattr(tree, 'shape'):
    out[prefix[:-1]] = np.asarray(tree, dtype=np.float32
                                  if np.asarray(tree).dtype.kind == 'f'
                                  else None)
  return out


def tapping(store, outer):
  """A haiku interceptor recording every module call's array output, keyed by
  module name and call order -- the same keys on both sides, so the first
  key that disagrees is the module that diverged."""
  import haiku as hk
  import jax
  counts = {}

  def interceptor(next_f, args, kwargs, context):
    out = next_f(*args, **kwargs)
    name = context.module.module_name
    if context.method_name == '__call__':
      n = counts.get(name, 0)
      counts[name] = n + 1
      # Only values of the OUTER trace: a scan body's (a layer_stack's) cannot
      # leave the scan. The stack's own output is tapped from its caller.
      leaves = [x for x in jax.tree_util.tree_leaves(out)
                if getattr(x, '_trace', None) is outer[0]]
      for i, x in enumerate(leaves):
        # jit returns dicts key-SORTED, so call order travels as its own leaf.
        key = '%s/%d/%d' % (name, n, i)
        store['taporder/' + key] = np.int32(len(store) // 2)
        store['tap/' + key] = x
    return out
  return hk.intercept_methods(interceptor)


def featurise(json_path, out_path):
  from alphafold3.common import folding_input
  from alphafold3.constants import chemical_components
  from alphafold3.data import featurisation

  with open(json_path) as f:
    fold_input = folding_input.Input.from_json(f.read())
  batch = featurisation.featurise_input(
      fold_input=fold_input,
      ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
      buckets=None)[0]
  np.savez(out_path, **{k: np.asarray(v) for k, v in batch.items()
                        if np.asarray(v).dtype.kind != 'O'})


def forward(in_path, out_path):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp

  z = np.load(in_path)
  batch = {k[len('batch/'):]: jnp.asarray(z[k]) for k in z.files
           if k.startswith('batch/')}
  pos = jnp.asarray(z['positions'])            # (num_tokens, atoms, 3)
  pos_noisy = jnp.asarray(z['positions_noisy'])
  sigma = jnp.asarray(z['noise_level'])

  cfg = af3_model.Model.Config()
  cfg.global_config.bfloat16 = str(z['bfloat16'])
  cfg.global_config.flash_attention_implementation = 'xla'
  cfg.num_recycles = int(z['num_recycles'])
  cfg.return_embeddings = True
  cfg.return_distogram = True

  class Injected(af3_model.Model):

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config):
      with tapping(taps, outer):
        denoised = self.diffusion_module(
            positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
            embeddings=embeddings, use_conditioning=True)
      mask = batch.predicted_structure_info.atom_mask
      return {'atom_positions': pos[None], 'mask': mask[None],
              'denoised': denoised[None]}

  taps, outer = {}, [None]

  @hk.transform
  def fwd(b):
    outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
    out = Injected(cfg)(b)
    return {**out, **taps}

  params = afp.get_model_haiku_params(model_dir=str(z['model_dir']))
  out = jax.jit(fwd.apply)(params, jax.random.PRNGKey(int(z['seed'])), batch)
  np.savez(out_path, **_flatten(jax.device_get(out)))


if __name__ == '__main__':
  sys.argv, (cmd, a, b) = sys.argv[:1], sys.argv[1:4]
  {'featurise': featurise, 'forward': forward}[cmd](a, b)
