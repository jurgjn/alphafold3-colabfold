"""Do the SHIPPED entry points agree with each other, for model X?

  PYTHONPATH=src:. python dev/oracles/shipped_path_check.py <model> [PDB]

The module gates call one module at a time with their own config, so they never
reach the paths a user's fold actually takes -- and since 2026-09-16 those paths
grew three alternatives to what the gates measure:

  staged     `staged.make_stages` drives the trunk one pass at a time (live mode,
             the colab notebook). Compared against the fused fori_loop: the trunk
             rng chain is the same, so the embeddings must agree.
  cached     `Model._sample_diffusion` builds the diffusion conditioning ONCE per
             fold and hands it to every step. Compared against one step that
             builds it inline -- the only path any vendor gate measures.
  float16    `half_dtype='float16'`, what a pre-Ampere card runs. Graded loosely
             (half precision is not parity) but it must not overflow.

This compares OURS to OURS. It does not replace a vendor gate; it says whether a
vendor gate's verdict on the inline path carries over to the paths users run.
Prints `__VERDICT PASS|FAIL` and exits non-zero on FAIL.
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _cmp(name, a, b, need):
  import af3_native_parity as P
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  if not (np.isfinite(a).all() and np.isfinite(b).all()):
    print('  %-28s NONFINITE  ours-a %s  ours-b %s' % (
        name, np.isfinite(a).all(), np.isfinite(b).all()))
    return False
  c, r = P.compare(a, b)
  g = P.grade(c, r)
  ok = g in need
  print('  %-28s corr %.6f  max|d|/rms %.2e  %s%s'
        % (name, c, r, g, '' if ok else '  <-- FAIL'))
  return ok


def main(argv):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp
  from alphafold3.model import staged
  import fold_check

  model_name = argv[0]
  pdb = argv[1] if len(argv) > 1 else os.path.expanduser('~/6MRR.pdb')
  seq, _ = fold_check.parse_ca(pdb)
  batch, cfg, model_dir = fold_check._fold_setup(model_name, seq)
  cfg.global_config.bfloat16 = 'none'
  cfg.num_recycles = 1
  cfg.return_embeddings = True
  params = afp.get_model_haiku_params(model_dir=model_dir)
  key = jax.random.PRNGKey(0)
  ok = True
  print('%s: %d residues, fp32, %d trunk passes'
        % (model_name, len(seq),
           af3_model.num_trunk_passes(cfg.num_recycles, model_name)))

  # --- staged trunk vs fused trunk
  @hk.transform
  def fused(b):
    return af3_model.Model(cfg)(b, structure=False)
  fo = jax.jit(fused.apply)(params, key, batch)
  st = staged.make_stages(cfg)
  carry, k = st['embed'](params, key, batch, None, None)
  for _ in range(af3_model.num_trunk_passes(cfg.num_recycles, model_name)):
    carry, k, _contacts = st['trunk'](params, key, batch, carry, k)
  ok &= _cmp('staged/single', carry['single'], fo['single_embeddings'],
             ('PARITY',))
  ok &= _cmp('staged/pair', carry['pair'], fo['pair_embeddings'], ('PARITY',))

  # --- cached vs inline diffusion conditioning, one step, injected inputs
  rng = np.random.default_rng(0)
  amask = np.asarray(batch['pred_dense_atom_mask'], np.float32)[..., None]
  sigma = jnp.float32(10.0)
  pos_noisy = jnp.asarray(rng.normal(size=amask.shape[:2] + (3,)) * 10
                          * amask, jnp.float32)
  out = {}

  class Injected(af3_model.Model):

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config,
                          diffusion_state=None):
      pc, ac = self.diffusion_module(
          positions_noisy=None, noise_level=jnp.zeros(()), batch=batch,
          embeddings=embeddings, use_conditioning=True, conditioning_only=True)
      out['cached'] = self.diffusion_module(
          positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
          embeddings=embeddings, use_conditioning=True,
          pair_cond=pc, atom_cond=ac)
      out['inline'] = self.diffusion_module(
          positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
          embeddings=embeddings, use_conditioning=True)
      m = batch.predicted_structure_info.atom_mask
      return {'atom_positions': out['cached'][None], 'mask': m[None]}

  @hk.transform
  def injected(b):
    Injected(cfg)(b)
    return dict(out)
  d = jax.jit(injected.apply)(params, key, batch)
  ok &= _cmp('denoise cached vs inline', d['cached'], d['inline'], ('PARITY',))

  # --- float16 trunk vs fp32. CPU has no F16_F16_F32 dot, and a check that
  # cannot run is reported, not passed: the driver runs this on the GPU.
  if jax.default_backend() == 'cpu':
    print('  float16                      NOT RUN on cpu')
    print('__VERDICT %s' % ('PASS' if ok else 'FAIL %s' % model_name))
    return 0 if ok else 1
  cfg16 = copy.deepcopy(cfg)
  cfg16.global_config.bfloat16 = 'all'
  cfg16.global_config.half_dtype = 'float16'

  @hk.transform
  def fused16(b):
    return af3_model.Model(cfg16)(b, structure=False)
  f16 = jax.jit(fused16.apply)(params, key, batch)
  # Half precision is not parity; LOOSE is the bar, NaN/inf is the failure.
  for k_ in ('single_embeddings', 'pair_embeddings'):
    ok &= _cmp('float16/' + k_, f16[k_], fo[k_], ('PARITY', 'CLOSE', 'LOOSE'))

  print('__VERDICT %s' % ('PASS' if ok else 'FAIL %s' % model_name))
  return 0 if ok else 1


if __name__ == '__main__':
  argv, sys.argv = sys.argv[1:], sys.argv[:1]   # tokamax parses argv lazily
  sys.exit(main(argv))
