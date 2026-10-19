"""chai-1 confidence head, ours against chai's OWN head at fp32, bisected by input path.

chai1_confidence_parity.py feeds our head chai's captured ConfidenceHead inputs
and compares against the captured (bf16) logits. This reruns chai's head on the
same inputs with its in-graph bf16 casts rewritten to fp32
(chai1_upstream_runner._fp32_graphs), so both sides are fp32, and then repeats
with one input path ablated on BOTH sides -- the single input repr, the trunk
single, the trunk pair, the coordinates -- to say which path carries a gap.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:dev/oracles \\
    python dev/oracles/chai1_confidence_bisect.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.expanduser(os.environ.get('CHAI_CONF_CAP', '~/chai_6mrr/conf_seam.npz'))
ORDER = ['token_single_input_repr', 'token_single_trunk_repr', 'token_pair_trunk_repr',
         'token_single_mask', 'atom_single_mask', 'atom_coords', 'token_reference_atom_index',
         'atom_token_index', 'atom_within_token_index']


def chai_head():
  import torch
  import chai1_upstream_runner as R
  m = torch.jit.load(os.path.expanduser('~/chai1_weights/models_v2/confidence_head.pt'), map_location='cpu')
  m.eval()
  R._fp32_graphs(m)
  fwd = m._c._get_method('forward_256')

  def run(f):
    args = []
    for k in ORDER:
      v = f[k]
      if k in ('token_single_mask', 'atom_single_mask'):
        args.append(torch.from_numpy(v).bool())
      elif k.endswith('_index'):
        args.append(torch.from_numpy(v).long())
      else:
        args.append(torch.from_numpy(v.astype(np.float32)))
    with torch.no_grad():
      return [np.asarray(o.float()) for o in fwd(*args)]
  return run


def ours_head(L):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  import fold_check
  from alphafold3.model import feat_batch, params as afp
  from alphafold3.model.components import haiku_modules as hm
  from alphafold3.model.network import confidence_head as ch

  seq, _ = fold_check.parse_ca(os.path.expanduser('~/6MRR.pdb'))
  batch, cfg, model_dir = fold_check._fold_setup('chai1', seq)
  fb = feat_batch.Batch.from_data_dict(batch)
  pb = fb.pseudo_beta_info.token_atoms_to_pseudo_beta
  full = afp.get_model_haiku_params(model_dir=model_dir)
  want = ('pae_logits', 'left_half_distance_logits')

  def run(f):
    rep = np.asarray(f['token_reference_atom_index'])[0].astype(int)[:L]
    coords = np.asarray(f['atom_coords'])[0]
    idxs = np.asarray(pb.gather_idxs).reshape(-1)[:L]
    dense = np.zeros(np.asarray(fb.ref_structure.mask).shape + (3,), np.float32)
    dense.reshape(-1, 3)[idxs] = coords[rep]
    emb = {'pair': jnp.asarray(f['token_pair_trunk_repr'][0, :L, :L].astype(np.float32)),
           'single': jnp.asarray(f['token_single_trunk_repr'][0, :L].astype(np.float32)),
           'target_feat': jnp.asarray(f['token_single_input_repr'][0, :L].astype(np.float32))}
    taps = {}
    orig = hm.Linear.__call__

    def tapped(self, x, *a, **kw):
      out = orig(self, x, *a, **kw)
      if self.name in want:
        taps[self.name] = out
      return out
    hm.Linear.__call__ = tapped
    try:
      def fwd():
        return ch.ConfidenceHead(cfg.heads.confidence, cfg.global_config)(
            dense_atom_positions=jnp.asarray(dense), embeddings=emb, seq_mask=fb.token_features.mask,
            token_atoms_to_pseudo_beta=pb, asym_id=fb.token_features.asym_id,
            token_features=fb.token_features, atom_name_chars=fb.ref_structure.atom_name_chars)
      t = hk.transform(fwd)
      init = t.init(jax.random.PRNGKey(0))
      p = {sc: {k: np.asarray((full.get('diffuser/' + sc) or full[sc])[k], np.float32) for k in init[sc]}
           for sc in init}
      if os.environ.get('SWAP_S2P'):
        # probe: the single->pair halves on the other token axis
        l = [sc for sc in p if sc.endswith('left_target_feat_project')][0]
        r = [sc for sc in p if sc.endswith('right_target_feat_project')][0]
        p[l], p[r] = p[r], p[l]
      t.apply(p, jax.random.PRNGKey(0))
    finally:
      hm.Linear.__call__ = orig
    pde = np.asarray(taps['left_half_distance_logits'])
    return np.asarray(taps['pae_logits']), pde + np.swapaxes(pde, 0, 1)
  return run


def cmp(name, a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  d = np.abs(a - b)
  print('  %-10s corr %.7f  max|d|/rms %.2e' % (name, np.corrcoef(a.ravel(), b.ravel())[0, 1],
                                                d.max() / np.sqrt((b ** 2).mean())))


def main():
  z = np.load(CAP)
  base = {k: np.asarray(z['in|' + k]) for k in ORDER}
  L = int(base['token_single_mask'].sum())
  chai, ours = chai_head(), ours_head(L)
  variants = {
      'none': {},
      'zero single input': {'token_single_input_repr': 0},
      'zero trunk single': {'token_single_trunk_repr': 0},
      'zero trunk pair': {'token_pair_trunk_repr': 0},
      'coords at origin': {'atom_coords': 0},
  }
  for name, ablate in variants.items():
    f = dict(base)
    for k in ablate:
      f[k] = np.zeros_like(base[k])
    c = chai(f)
    o_pae, o_pde = ours(f)
    print('ablate: %s' % name)
    cmp('pae', o_pae, c[0][0, :L, :L])
    cmp('pde', o_pde, c[1][0, :L, :L])


if __name__ == '__main__':
  sys.argv = sys.argv[:1]
  main()
