"""A ported model against its STOCK vendor, whole model, every input modality.

  ~/venv/bin/python dev/oracles/vendor_native_parity.py MODEL [CASE ...] [--recycles N]

Both sides run their SHIPPED entry points on the same input. The vendor side is
a per-family runner (VENDORS below) executed in the vendor's own environment:
its own featuriser, its own model loader, its own top-level forward. Ours: our
featurisation with the model's spec, `Model.__call__`. The only things either
side is handed are RANDOM DRAWS -- the vendor's reference-conformer draw
(`ref_pos`), the samples the confidence head scores, and one (noised
coordinates, sigma) the denoiser runs on -- so no constructor argument, flag or
feature is chosen by this file. That is the point: the hand-built adapters this
replaces as the gate of record each re-derived the vendor's setup and could get
it wrong silently (boltz2's fix_sym_check, 2026-10-18).

RUNNER CONTRACT (see boltz2_upstream_runner.py, protenix_upstream_runner.py):
  featurise MODEL CASE.json DIR      -> DIR/layout.npz: atom_name, atom_token,
                                        atom_pad, token_pad, feat.ref_pos,
                                        feat.ref_space_uid
  forward MODEL DIR IN.npz OUT.npz   <- positions, positions_noisy, noise_level,
                                        recycling_steps, precision (all in the
                                        vendor's flat atom order)
                                     -> s_inputs, s, z, distogram_logits,
                                        denoised, pae, pde, plddt

Graded, per case: s_inputs, trunk single/pair after recycling, distogram
logits, one denoise step, pLDDT / PAE / PDE. Atoms are aligned by (token, atom
name), so an atom one side has and the other does not is REPORTED, never
silently dropped. Prints `__VERDICT PASS|FAIL`.
"""
import json
import os
import subprocess
import sys
import tempfile

# Precision is not the model: XLA's default GPU matmul is TF32-class, which
# alone reads the boltz2 denoise step at 2e-2 where 'highest' reads 3e-5.
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af3_native_parity as P   # noqa: E402  (cases, compare, grade)

_H = os.path.expanduser('~')
# model -> (interpreter, PYTHONPATH for the vendor's tree or None, runner)
VENDORS = {
    'boltz2': (_H + '/venv_boltz/bin/python', None, 'boltz2_upstream_runner.py'),
    'protenix2': (_H + '/venv_protenix/bin/python', _H + '/protenix',
                  'protenix_upstream_runner.py'),
    'protenix1': (_H + '/venv_protenix/bin/python', _H + '/protenix',
                  'protenix_upstream_runner.py'),
    # OpenDDE v1.1.0 in protenix's venv: every pin in its pyproject (torch
    # 2.7.1, numpy 2.4.1, rdkit 2025.9.3, biotite 1.4.0, ...) is met exactly.
    'opendde': (_H + '/venv_protenix/bin/python', _H + '/OpenDDE',
                'opendde_upstream_runner.py'),
    # OpenFold3: ONE runner, the tree picks the release. openbind0 is v0.5.0
    # (worktree ~/openfold-3-v050) -- main loads its tensors but runs the other
    # triangle-attention convention (run_all_parity.sh). ~/venv_of3 serves both:
    # the two trees' conda locks are identical (pins in
    # ~/venv_of3/requirements-of3.txt; deepspeed/cuequivariance left out).
    # openfold3 = stock release 0.4.5 (a4b08032), the newest that loads the
    # preview-2 checkpoint; origin/main (v0.5.0-155-g8090aa69) refuses it.
    'openfold3': (_H + '/venv_of3/bin/python', _H + '/openfold-3-045',
                  'openfold3_upstream_runner.py'),
    'openbind0': (_H + '/venv_of3/bin/python', _H + '/openfold-3-v050',
                  'openfold3_upstream_runner.py'),
    # IntelliFold v2.0.4's native PyTorch stack. Its pyproject pins only the
    # JAX wrapper, so ~/venv_intellifold (uv, python 3.11) carries the pins of
    # the last wheel that shipped the torch stack (pypi/intellifold-2.0.2:
    # torch 2.6.0, numpy 1.24.0, rdkit 2024.3.2, accelerate 1.1.1, ...) minus
    # deepspeed, a CUDA-build kernel extra the stack only uses when asked.
    'intellifold2': (_H + '/venv_intellifold/bin/python', _H + '/IntelliFold',
                     'intellifold2_upstream_runner.py'),
    # RoseTTAFold3 (foundry @ aad357b). ~/venv_rf3 (uv, python 3.12) pins
    # foundry's pyproject deps under its own ci-constraints.txt (torch 2.7.1,
    # rdkit 2025.3.6; atomworks 2.2.1) in ~/venv_rf3/requirements-rf3.txt,
    # minus the `rf3` extra (cuequivariance: CUDA-build fused kernels).
    'rosettafold3': (_H + '/venv_rf3/bin/python',
                     _H + '/foundry_rf3/src:' + _H + '/foundry_rf3/models/rf3/src',
                     'rosettafold3_upstream_runner.py'),
}


VENDOR_NA_EXIT = 3


class VendorNA(Exception):
  """The stock vendor has no way to take this input (runner exit 3)."""


def _vendor(model, *args):
  py, path, runner = VENDORS[model]
  env = {k: v for k, v in os.environ.items()
         if k not in ('PYTHONPATH', 'JAX_PLATFORMS')}
  # -I isolates from OUR tree; a vendor that is a source checkout instead gets
  # exactly that checkout on its path (-P: not the runner's own directory).
  cmd = [py, '-I'] if path is None else [py, '-P']
  if path is not None:
    env['PYTHONPATH'] = path
  if os.environ.get('VENDOR_CPU'):
    env['CUDA_VISIBLE_DEVICES'] = ''
  r = subprocess.run(cmd + [os.path.join(HERE, runner), args[0], model,
                            *args[1:]],
                     capture_output=True, text=True, env=env,
                     cwd=tempfile.gettempdir())
  # Exit 3 = the STOCK vendor cannot express this input at all (openfold3 has
  # no bond input, so `covalent` is N/A there): reported as N/A, never PASS,
  # and never confused with a crash, which stays a FAILURE.
  if r.returncode == VENDOR_NA_EXIT:
    raise VendorNA(r.stderr.strip().splitlines()[-1] if r.stderr.strip()
                   else 'input not expressible in the vendor')
  if r.returncode:
    raise RuntimeError('%s %s failed:\n%s' % (model, args[0], r.stderr[-4000:]))


def _s_inputs_to_ours(model, x):
  """The vendor's s_inputs in OUR column order, where the layouts differ.

  The OF3-derived families (protenix, opendde, ...) are 449 wide with their own
  restype order; ours is 447 = their two 32-class one-hots permuted to AF3's
  order, the deletion mean, and the 384 atom-pooled channels -- the same map
  denoise_parity and conditioning_parity build their inputs with.
  """
  if x.shape[-1] == 449:
    from converters.openfold3 import _AF3_TO_OF3_AATYPE as remap
    # rosettafold3 (atomworks' alphabet) swaps G/C and DG/DC relative to OF3's
    # (converters/rosettafold3.py) -- its own remap, or every nucleic s_inputs
    # row reads as a disagreement.
    if model == 'rosettafold3':
      from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as remap
    idx = np.concatenate([384 + np.asarray(remap), 416 + np.asarray(remap),
                          [448], np.arange(384)])
    return x[..., idx]
  return x


def _names(chars):
  """AF3/boltz atom-name chars (ord - 32) -> stripped strings."""
  return np.array([''.join(chr(int(c) + 32) for c in row).strip()
                   for row in chars.reshape(-1, chars.shape[-1])]
                  ).reshape(chars.shape[:-1])


def featurise_ours(model, j, recycles, bf16):
  import fold_check
  from alphafold3.common import folding_input
  fi = folding_input.Input.from_json(json.dumps(j))
  # MODEL_DIR (fold_check's convention): an alternative blob for the same
  # model, e.g. intellifold2 converted with IF2_FP32_BLOB=1, to tell its bf16
  # trunk storage policy from a port bug.
  batch, cfg, model_dir = fold_check._fold_setup(
      model, None, model_dir=os.environ.get('MODEL_DIR'), chains=list(fi.chains),
      bonded_atom_pairs=fi.bonded_atom_pairs, seed=1)
  cfg.global_config.bfloat16 = bf16
  cfg.num_recycles = recycles
  cfg.return_embeddings = True
  cfg.return_distogram = True
  return batch, fold_check.LAST_RAW_BATCH, cfg, model_dir


def align(raw, lay):
  """-> (ours (token, slot) for each boltz atom or (-1,-1), report rows)."""
  ours_names = _names(np.asarray(raw['ref_atom_name_chars']))
  ours_mask = np.asarray(raw['ref_mask']).astype(bool)
  n_tok_o = ours_mask.shape[0]
  tp = lay['token_pad']
  rows = []
  if int(tp.sum()) != n_tok_o:
    rows.append(('align', 'token count ours %d boltz %d' % (n_tok_o,
                                                            int(tp.sum())),
                 'BAD'))
  where = {}
  for t in range(n_tok_o):
    for s in np.flatnonzero(ours_mask[t]):
      where[(t, ours_names[t, s])] = (t, s)
  idx = np.full((len(lay['atom_name']), 2), -1)
  missing_ours = []
  for i, (nm, t, real) in enumerate(zip(lay['atom_name'], lay['atom_token'],
                                        lay['atom_pad'])):
    if not real:
      continue
    if (t, nm) in where:
      idx[i] = where.pop((t, nm))
    else:
      missing_ours.append('%d:%s' % (t, nm))
  # SECOND PASS, for atoms with no canonical name -- a SMILES ligand's are
  # invented independently on each side (boltz numbers them C22..., we C1...).
  # Both parse the same SMILES with RDKit and give each ligand atom its own
  # token, so within a token a single unmatched atom on each side is the same
  # atom if the ELEMENT agrees. Reported, so a name convention is never mistaken
  # for agreement on atoms.
  if missing_ours:
    o_el = np.asarray(raw['ref_element'])
    b_el = (lay['feat.ref_element'].argmax(-1)
            if lay['feat.ref_element'].ndim == 2 else lay['feat.ref_element'])
    left_o = {}
    for (t, nm), ts in where.items():
      left_o.setdefault(t, []).append(ts)
    still, by_elem = [], 0
    for item in missing_ours:
      t, nm = item.split(':', 1)
      t = int(t)
      i = next(k for k, (n2, t2, r2) in enumerate(zip(
          lay['atom_name'], lay['atom_token'], lay['atom_pad']))
               if r2 and t2 == t and n2 == nm)
      cand = left_o.get(t, [])
      if len(cand) == 1 and int(o_el[cand[0]]) == int(b_el[i]):
        idx[i] = cand[0]
        by_elem += 1
        where.pop((t, _names(np.asarray(raw['ref_atom_name_chars'])[cand[0]][None])[0]), None)
        left_o[t] = []
      else:
        still.append(item)
    if by_elem:
      rows.append(('align', '%d atom(s) matched by token+element, names differ '
                   '(SMILES naming)' % by_elem, 'INFO'))
    missing_ours = still
  if missing_ours:
    rows.append(('align', 'boltz atoms ours lacks: %s' % ' '.join(
        missing_ours[:8]), 'BAD'))
  if where:
    rows.append(('align', 'our atoms boltz lacks: %s' % ' '.join(
        '%d:%s' % k for k in list(where)[:8]), 'INFO'))
  return idx, rows


def forward_ours(batch, cfg, model_dir, pos, pos_noisy, sigma):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp
  pos, pos_noisy = jnp.asarray(pos), jnp.asarray(pos_noisy)
  sigma = jnp.float32(sigma)

  class Injected(af3_model.Model):

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config,
                          diffusion_state=None):
      pc, ac = self.diffusion_module(
          positions_noisy=None, noise_level=jnp.zeros(()), batch=batch,
          embeddings=embeddings, use_conditioning=True, conditioning_only=True)
      den = self.diffusion_module(
          positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
          embeddings=embeddings, use_conditioning=True,
          pair_cond=pc, atom_cond=ac)
      m = batch.predicted_structure_info.atom_mask
      return {'atom_positions': pos[None], 'mask': m[None],
              'denoised': den[None]}

  from af3_upstream_runner import tapping
  taps, outer = {}, [None]

  @hk.transform
  def fwd(b):
    outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
    if os.environ.get('B2_KEEP'):
      with tapping(taps, outer):
        out = Injected(cfg)(b)
      return {**out, **taps}
    return Injected(cfg)(b)
  params = afp.get_model_haiku_params(model_dir=model_dir)
  out = jax.jit(fwd.apply)(params, jax.random.PRNGKey(0), batch)
  if os.environ.get('B2_KEEP'):
    # ONE trunk pass through the staged entry point, tapped: the fused path
    # runs the trunk inside a fori_loop, whose values cannot leave the loop.
    from alphafold3.model import staged as _staged
    st_taps, st_outer = {}, [None]

    def _trunk_tapped(b):
      st_outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
      m = af3_model.Model(cfg)
      carry, k = m(b, stage='embed')
      # Every pass but the last untapped; the last one tapped, so a gap that
      # only appears with recycling is localised on the pass where it appears.
      for _ in range(int(os.environ.get('B2_TAP_PASS', 1)) - 1):
        carry, k, _ = m(b, recycle_carry=carry, key=k, stage='trunk')
      with tapping(st_taps, st_outer):
        carry, k, _ = m(b, recycle_carry=carry, key=k, stage='trunk')
      return {**{'staged/' + kk: v for kk, v in carry.items()
                 if hasattr(v, 'shape')}, **st_taps}
    tt = hk.transform(_trunk_tapped)
    out = {**out, **jax.jit(tt.apply)(params, jax.random.PRNGKey(0), batch)}
  from af3_upstream_runner import _flatten
  return _flatten(jax.device_get(out))


def run_case(model, name, j, recycles, bf16):
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    cj = os.path.join(tmp, 'case.json')
    with open(cj, 'w') as f:
      json.dump(j, f)
    bdir = os.path.join(tmp, 'vendor')
    _vendor(model, 'featurise', cj, bdir)
    lay = dict(np.load(os.path.join(bdir, 'layout.npz')))
    batch, raw, cfg, model_dir = featurise_ours(model, j, recycles, bf16)
    idx, arows = align(raw, lay)
    rows += arows
    ok = idx[:, 0] >= 0

    # REF_POS IS A RANDOM DRAW ON BOTH SIDES, so it is injected like the
    # coordinates: boltz picks one of the residue's stored conformers at random
    # and applies a random roto-translation per ref_space_uid group
    # (featurizerv2.py:1494); ours takes the CCD ideal geometry, centred. Our
    # batch gets boltz's exact draw. What the two DISTRIBUTIONS differ in is
    # reported, not hidden: the largest within-group distance difference.
    import jax.numpy as jnp
    b_ref = lay['feat.ref_pos']
    rp = np.array(batch['ref_pos'])
    o_ref = rp[idx[ok, 0], idx[ok, 1]]
    uid = lay['feat.ref_space_uid'][ok]
    geo = max((np.abs(np.linalg.norm(o_ref[uid == u][:, None] - o_ref[uid == u][None], axis=-1)
                      - np.linalg.norm(b_ref[ok][uid == u][:, None] - b_ref[ok][uid == u][None], axis=-1)).max()
               for u in np.unique(uid)), default=0.0)
    rows.append(('feat', 'conformer geometry (ours CCD ideal vs boltz stored '
                 'draw): max within-residue distance diff %.2f A' % geo, 'INFO'))
    rp[idx[ok, 0], idx[ok, 1]] = b_ref[ok]
    batch = dict(batch, ref_pos=jnp.asarray(rp))
    # OPENDDE DIFFUSES ON STRUCTURAL TOKENS: our batch carries a second,
    # structural-token copy of the atom features under 'struct/' (the diffusion
    # module's atom encoder reads struct/ref_pos), and the sampler's positions
    # and the denoiser's output live in that (struct token, slot) layout. The
    # atoms are the same ones; residue_atom_gather[r, j] is the flat struct
    # index of residue atom (r, j). So the injected draw goes into both copies,
    # and positions cross between layouts through the gather -- a bijection on
    # real atoms, checked here.
    gather = batch.get('structbook/residue_atom_gather')
    if gather is not None:
      gather = np.asarray(gather)
      gv = gather >= 0
      if not (np.array_equal(gv, np.asarray(raw['pred_dense_atom_mask']).astype(bool))
              and len(np.unique(gather[gv])) == gv.sum()):
        rows.append(('align', 'structural gather is not a bijection on real '
                     'atoms', 'BAD'))
      s_shape = np.shape(batch['struct/pred_dense_atom_mask'])

      def to_struct(x):
        out = np.zeros((s_shape[0] * s_shape[1],) + x.shape[2:], x.dtype)
        out[gather[gv]] = x[gv]
        return out.reshape(s_shape + x.shape[2:])
      batch = dict(batch, **{'struct/ref_pos': jnp.asarray(to_struct(rp))})

    rng = np.random.default_rng(int(os.environ.get('INJECT_SEED', 0)))
    dmask = np.asarray(raw['pred_dense_atom_mask']).astype(np.float32)[..., None]
    pos = (rng.normal(size=dmask.shape[:2] + (3,)) * 8 * dmask).astype(np.float32)
    sigma = np.float32(os.environ.get('INJECT_SIGMA', 10.0))
    noisy = (pos + rng.normal(size=pos.shape) * sigma * dmask).astype(np.float32)
    to_b = lambda x: np.where(ok[:, None], x[idx[:, 0], idx[:, 1]], 0
                              ).astype(np.float32)
    inp = os.path.join(tmp, 'in.npz')
    np.savez(inp, positions=to_b(pos), positions_noisy=to_b(noisy),
             noise_level=sigma, recycling_steps=recycles,
             precision='bf16' if bf16 != 'none' else 'fp32')
    _vendor(model, 'forward', bdir, inp, os.path.join(tmp, 'b.npz'))
    b = dict(np.load(os.path.join(tmp, 'b.npz')))
    if os.environ.get('B2_KEEP'):
      import shutil
      keep = os.path.join(os.environ['B2_KEEP'], model, name)
      shutil.rmtree(keep, ignore_errors=True)
      shutil.copytree(bdir, keep)
      np.savez(os.path.join(keep, 'boltz_out.npz'), **b)
      np.savez(os.path.join(keep, 'ours_raw.npz'), **{
          k: np.asarray(v) for k, v in raw.items()
          if np.asarray(v).dtype.kind != 'O'})
  if gather is not None:
    ours = forward_ours(batch, cfg, model_dir, to_struct(pos), to_struct(noisy),
                        sigma)
    den = np.asarray(ours['diffusion_samples/denoised'])
    flat = den.reshape(den.shape[0], -1, 3)
    ours['diffusion_samples/denoised'] = np.where(
        gv[None, :, :, None], flat[:, np.where(gv, gather, 0)], 0.0)
  else:
    ours = forward_ours(batch, cfg, model_dir, pos, noisy, sigma)

  n = int(lay['token_pad'].sum())
  pairs = [
      ('s_inputs (target_feat)', ours['target_feat'],
       _s_inputs_to_ours(model, b['s_inputs'][:n])),
      ('trunk single', ours['single_embeddings'], b['s'][:n]),
      ('trunk pair', ours['pair_embeddings'], b['z'][:n, :n]),
      ('distogram logits', ours['distogram/distogram'],
       b['distogram_logits'][:n, :n].reshape(n, n, -1)),
      ('pae', ours['full_pae'][0], b['pae'][:n, :n]),
      ('pde', ours['full_pde'][0], b['pde'][:n, :n]),
  ]
  # Atom-level outputs, through the alignment.
  den_o = ours['diffusion_samples/denoised'][0][idx[ok, 0], idx[ok, 1]]
  pairs.append(('denoise step', den_o, b['denoised'].reshape(-1, 3)[ok]))
  # boltz2 reports pLDDT per TOKEN (token_level_confidence); ours carries the
  # token's value on each of its atoms, so read it at the first real slot.
  pl_b = b['plddt'].reshape(-1)
  if pl_b.shape[0] == len(ok):
    pl_o, pl_b = ours['predicted_lddt'][0][idx[ok, 0], idx[ok, 1]], pl_b[ok]
  else:
    first = np.asarray(raw['pred_dense_atom_mask']).astype(bool).argmax(1)
    pl_o, pl_b = ours['predicted_lddt'][0][np.arange(n), first], pl_b[:n]
  pairs.append(('plddt', pl_o, pl_b * (100.0 if pl_o.max() > 1.5 else 1.0)))
  if os.environ.get('B2_KEEP'):
    np.savez(os.path.join(os.environ['B2_KEEP'], model, name, 'ours_out.npz'),
             **{k: np.asarray(v) for k, v in ours.items()})
  for label, a, r in pairs:
    a, r = np.asarray(a, np.float32), np.asarray(r, np.float32)
    if a.shape != r.shape:
      rows.append(('model', '%s shape ours %s boltz %s' % (label, a.shape,
                                                          r.shape), 'BAD'))
      continue
    c, ratio = P.compare(a, r)
    rows.append(('model', '%s corr %.6f max|d|/rms %.2e' % (label, c, ratio),
                 P.grade(c, ratio)))
  return rows


def main(argv):
  rec = 1
  if '--recycles' in argv:
    rec = int(argv[argv.index('--recycles') + 1])
  pos = [a for a in argv if not a.startswith('--') and not a.isdigit()]
  model, names = pos[0], pos[1:]
  if model not in VENDORS:
    raise SystemExit('no vendor runner for %r; have %s' % (model, sorted(VENDORS)))
  allc = P.cases()
  names = names or [n for n in allc if n != 'template']
  failed = []
  for n in names:
    print('== %s %s' % (model, n), flush=True)
    try:
      rows = run_case(model, n, allc[n], rec, 'none')
    except VendorNA as e:   # the vendor cannot take this input: N/A, not FAIL
      print('  %s: N/A (%s)' % (n, e), flush=True)
      continue
    except Exception as e:  # a crash is a FAILURE, never a skip
      import traceback
      traceback.print_exc()
      print('  ERROR %s: %s' % (type(e).__name__, str(e).splitlines()[0][:200]))
      failed.append(n)
      continue
    for stage, msg, g in rows:
      print('  %-6s %-7s %s' % (stage, g, msg))
    bad = [r for r in rows if r[2] not in ('PARITY', 'CLOSE', 'INFO')]
    print('  %s: %s' % (n, 'PASS' if not bad else 'FAIL (%d rows)' % len(bad)),
          flush=True)
    if bad:
      failed.append(n)
  print('__VERDICT %s %s' % ('FAIL' if failed else 'PASS', ' '.join(failed)))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
  # the repo root: converters/ (fold_check's parse_ca, rf3's alphabet remap)
  sys.path.insert(0, os.path.join(HERE, '..', '..'))
  argv, sys.argv = sys.argv[1:], sys.argv[:1]
  sys.exit(main(argv))
