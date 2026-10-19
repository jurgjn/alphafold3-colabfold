"""chai-1 native-vs-ours parity on a real fold input: featurised inputs, token embedder, trunk, and the fold.

chai ships TorchScript only, so nothing inside a module can be hooked. Native's exported components are called in
sequence by `run_folding_on_context` (feature embedding, bond projection, token embedder, trunk x recycles, diffusion,
confidence); `native` swaps `load_exported` for a recorder and keeps every tensor that crosses between them, plus the
collated raw features and inputs. `ours` featurises the same AF3 json exactly as ColabFold does
(colabfold.alphafold3.models.featurise) and catches the batch, the MSA stream, s_init, z_init and each trunk pass.
`compare` lines them up feature by feature; `fold` runs a ColabFold-equivalent fold and scores it against native's
samples with the gauntlet's own scorer. `--inject` substitutes native inputs (profile, deletion_mean, msa, ref_pos) to
price each difference alone.

  # native, in chai's venv (gpu001: . /data/milot/uplift/chai1/env.sh), inputs from the gauntlet's chai_input.py
  python -s chai1_trunk_parity.py native --fasta X.fasta --msa-dir D --work W --out nat.npz [--trunk-only]
  # ours, in a jax env with alphafold3 + colabfold importable
  python chai1_trunk_parity.py ours --json input.json --model-dir DIR --out ours.npz [--inject ref_pos --native nat.npz]
  python chai1_trunk_parity.py compare nat.npz ours.npz --control nat2.npz [--per-token]
  python chai1_trunk_parity.py fold --json input.json --model-dir DIR --out-dir O --native-result result.json

Read the controls: chai-lab against itself is not bit-exact (s_init 0.999998, pass-0 single 0.99997, final pair
0.9999), and our bf16 against our fp32 (`--bf16 none --xla`) sits at the same floor. A pass-0 single at 0.994 is a
defect, not noise: it was the MSA block's transition run in sequence.
"""
import argparse
import os
import sys

import numpy as np

# raw TOKEN-level features of chai's stream that ours has to reproduce
_SAVE_EMB_MSA_ROWS = 64


def _np(t):
  import torch
  if isinstance(t, torch.Tensor):
    t = t.detach()
    if t.dtype == torch.bfloat16:
      t = t.float()
    return t.cpu().numpy()
  return np.asarray(t)


def native(args):
  """Run chai-lab's own inference with a recorder around every exported component."""
  from pathlib import Path

  import torch
  import chai_lab.chai1 as C

  saved = {}
  n_tok = [None]

  def keep(key, t, crop=None):
    a = _np(t)
    if crop is not None:
      a = crop(a)
    if a.dtype == np.float64:
      a = a.astype(np.float32)
    saved[key] = a

  class Collate(C.Collate):
    def __call__(self, contexts):
      batch = super().__call__(contexts)
      for k, v in batch['inputs'].items():
        if isinstance(v, torch.Tensor) and v.numel() < 3e8:
          keep(f'inp/{k}', v)
      n_tok[0] = int(batch['inputs']['token_exists_mask'].sum())
      return batch

  C.Collate = Collate
  calls = {}

  def tok_crop(a, axes):
    n = n_tok[0]
    sl = [slice(None)] * a.ndim
    for ax in axes:
      sl[ax] = slice(0, n)
    return a[tuple(sl)]

  class Recorder:
    def __init__(self, name, inner):
      self.name, self.inner = name, inner

    def forward(self, crop_size, **kw):
      out = self.inner.forward(crop_size=crop_size, **kw)
      i = calls.get(self.name, 0)
      calls[self.name] = i + 1
      if self.name == 'feature_embedding.pt':
        for k, v in kw.items():
          if isinstance(v, torch.Tensor):
            keep(f'feat/{k}', v)
        for k, v in out.items():
          if k == 'MSA':
            keep('emb/MSA', v[:, :_SAVE_EMB_MSA_ROWS])
          elif k == 'TOKEN_PAIR':
            keep('emb/TOKEN_PAIR', v.half(), lambda a: tok_crop(a, (1, 2)))
          else:
            keep(f'emb/{k}', v)
      elif self.name == 'bond_loss_input_proj.pt':
        keep('bond/input', kw['input'])
      elif self.name == 'token_embedder.pt':
        s_trunk, s_struct, z_init = out
        keep('tie/s_trunk', s_trunk, lambda a: tok_crop(a, (1,)))
        keep('tie/s_struct', s_struct, lambda a: tok_crop(a, (1,)))
        keep('tie/z_init', z_init.half(), lambda a: tok_crop(a, (1, 2)))
        keep('tie/token_pair_input_feats', kw['token_pair_input_feats'].half(), lambda a: tok_crop(a, (1, 2)))
      elif self.name == 'trunk.pt':
        s, z = out
        keep(f'trunk/single_{i}', s, lambda a: tok_crop(a, (1,)))
        if args.all_pairs or i == args.recycles - 1:
          keep(f'trunk/pair_{i}', z.half(), lambda a: tok_crop(a, (1, 2)))
        if i == args.recycles - 1 and args.trunk_only:
          raise _Stop()
      elif self.name == 'confidence_head.pt':
        keep(f'conf/plddt_logits_{i}', out[2] if len(out) > 2 else out[-1])
      return out

  class _Stop(Exception):
    pass

  orig = C.load_exported
  C.load_exported = lambda key, device: Recorder(key, orig(key, device))
  out_dir = Path(args.work)
  out_dir.mkdir(parents=True, exist_ok=True)
  import shutil
  for p in out_dir.iterdir():
    shutil.rmtree(p) if p.is_dir() else p.unlink()
  ctx = C.make_all_atom_feature_context(
      Path(args.fasta), output_dir=out_dir, use_esm_embeddings=False,
      msa_directory=Path(args.msa_dir) if args.msa_dir else None,
      constraint_path=Path(args.constraints) if args.constraints else None)
  try:
    cands = C.run_folding_on_context(
        ctx, output_dir=out_dir, num_trunk_recycles=args.recycles, num_diffn_timesteps=args.steps,
        num_diffn_samples=args.samples, seed=args.seed, device=torch.device('cuda:0'), low_memory=True)
    for k in ('plddt', 'ptm_scores', 'aggregate_score'):
      v = getattr(cands, k, None)
      if v is not None and not hasattr(v, 'ptm'):
        saved[f'out/{k}'] = np.asarray([_np(x).astype(np.float32) for x in v]) if isinstance(v, list) else _np(v)
  except _Stop:
    pass
  saved['n_tokens'] = np.asarray(n_tok[0])
  np.savez(args.out, **saved)
  print(f'wrote {len(saved)} arrays to {args.out}; {n_tok[0]} tokens')


def _inject(ex, nat, what):
  """Replace our featurised inputs with native's, to price each difference on its own."""
  n = int(nat['n_tokens'])
  chai_to_ours = np.full(33, 31)
  chai_to_ours[_OURS_TO_CHAI[:31]] = np.arange(31)
  for w in what:
    if w == 'profile':
      ex['profile'] = nat['feat/MSAProfile'][0, :n][:, _OURS_TO_CHAI[:31]].astype(np.float32)
    elif w == 'deletion_mean':
      ex['deletion_mean'] = nat['feat/MSADeletionMean'][0, :n, 0].astype(ex['deletion_mean'].dtype)
    elif w == 'msa':
      live = np.flatnonzero(nat['inp/msa_mask'][0, :, :n].any(1))
      depth = max(ex['msa'].shape[0], live.size)
      msa = np.full((depth, n), 21, ex['msa'].dtype)
      dm = np.zeros((depth, n), np.float32)
      mask = np.zeros((depth, n), bool)
      msa[:live.size] = chai_to_ours[nat['inp/msa_tokens'][0, live, :n]]
      dm[:live.size] = nat['inp/msa_deletion_matrix'][0, live, :n]
      mask[:live.size] = nat['inp/msa_mask'][0, live, :n]
      ex['msa'], ex['deletion_matrix'], ex['msa_mask'] = msa, dm, mask
      ex['num_alignments'] = np.asarray(live.size, ex['num_alignments'].dtype)
    elif w == 'ref_pos':
      ati, exm = nat['inp/atom_token_index'][0], nat['inp/atom_exists_mask'][0]
      names, pos = nat['feat/AtomNameOneHot'][0], nat['inp/atom_ref_pos'][0]
      rp = np.array(ex['ref_pos'])
      for t in range(n):
        ia = np.flatnonzero(exm & (ati == t))
        by_name = {tuple(names[i]): pos[i] for i in ia}
        for a in np.flatnonzero(ex['ref_mask'][t]):
          key = tuple(ex['ref_atom_name_chars'][t, a])
          if key in by_name:
            rp[t, a] = by_name[key]
      ex['ref_pos'] = rp
    else:
      raise ValueError(f'nothing to inject called {w}')
  print('injected native ' + ', '.join(what))


def _no_glu_kernel(cfg, seen=None):
  """use_glu_kernel off everywhere in a config tree: XLA ops only."""
  seen = set() if seen is None else seen
  if id(cfg) in seen or not hasattr(cfg, '__dict__'):
    return
  seen.add(id(cfg))
  for k, v in list(vars(cfg).items()):
    if k == 'use_glu_kernel':
      setattr(cfg, k, False)
    elif hasattr(v, '__dict__') and not isinstance(v, type):
      _no_glu_kernel(v, seen)


def _patch_params(params, chain_entity=None, token_dx=None):
  """Rows a blob may predate, from .npy files: chai1_chain_entity (converters.chai1.chain_entity_rows) and
  chai1_token_dx (ResidueType DX - RX through chai's TOKEN projection)."""
  import jax.numpy as jnp
  params = dict(params)
  if chain_entity:
    params['diffuser/evoformer'] = {**params.get('diffuser/evoformer', {}),
                                    'chai1_chain_entity': jnp.asarray(np.load(chain_entity))}
  if token_dx:
    params['diffuser'] = {**params.get('diffuser', {}), 'chai1_token_dx': jnp.asarray(np.load(token_dx))}
  return params


def ours(args):
  """Featurise like ColabFold, run our chai1 trunk, and keep the batch and the stream at each seam."""
  import haiku as hk
  import jax
  from alphafold3.common import folding_input
  from alphafold3.model import model as af3_model, params as afp
  from alphafold3.model.network import evoformer, featurization
  from colabfold.alphafold3 import models as cf_models

  fold_input = folding_input.Input.from_json(open(args.json).read())
  fold_input = folding_input.Input(name=fold_input.name, chains=fold_input.chains, rng_seeds=[args.seed],
                                   bonded_atom_pairs=fold_input.bonded_atom_pairs, user_ccd=fold_input.user_ccd)
  ex = cf_models.featurise(fold_input, 'chai1', args.model_dir, buckets=None)[0]
  if args.inject:
    _inject(ex, np.load(args.native), args.inject.split(','))
  saved = {}
  for k, v in ex.items():
    v = np.asarray(v)
    if v.dtype != object and v.size < 3e8:
      saved[f'batch/{k}'] = v

  cfg = cf_models.make_config('chai1', args.recycles, 5)
  if args.bf16:
    cfg.global_config.bfloat16 = args.bf16
  if args.xla:
    cfg.global_config.flash_attention_implementation = 'xla'
    _no_glu_kernel(cfg)
  caught = {}

  def tap(name, x, half=False):
    def put(v):
      v = np.asarray(v, np.float16 if half else np.float32)
      caught.setdefault(name, []).append(v)
    jax.debug.callback(put, x)

  patches = []

  def patch(obj, name, fn):
    patches.append((obj, name, getattr(obj, name)))
    setattr(obj, name, fn(getattr(obj, name)))

  def msa_feat(orig):
    def f(*a, **kw):
      out = orig(*a, **kw)
      tap('msa_feat', out[:_SAVE_EMB_MSA_ROWS])
      return out
    return f

  def target_feat(orig):
    def f(*a, **kw):
      out = orig(*a, **kw)
      if isinstance(out, tuple):
        tap('s_trunk', out[0])
        tap('s_struct', out[1])
      else:
        tap('s_trunk', out)
      return out
    return f

  def rel_enc(orig):
    def f(self, *a, **kw):
      out = orig(self, *a, **kw)
      tap('z_init', out, half=True)
      return out
    return f

  def evo_call(orig):
    def f(self, *a, **kw):
      out = orig(self, *a, **kw)
      tap('single', out['single'])
      tap('pair', out['pair'], half=True)
      return out
    return f

  patch(featurization, 'create_msa_feat', msa_feat)
  patch(af3_model, 'create_target_feat_embedding', target_feat)
  patch(evoformer.Evoformer, '_relative_encoding', rel_enc)
  patch(evoformer.Evoformer, '__call__', evo_call)

  @hk.transform
  def fwd(b):
    return af3_model.Model(cfg)(b, structure=False)

  from alphafold3.model.components import utils as af3_utils
  params = _patch_params(afp.get_model_haiku_params(model_dir=args.model_dir), args.chain_entity, args.token_dx)
  ex = af3_utils.remove_invalidly_typed_feats(ex)
  try:
    out = jax.jit(fwd.apply)(params, jax.random.PRNGKey(args.seed), ex)
    jax.block_until_ready(out)
    jax.effects_barrier()
  finally:
    for obj, name, orig in reversed(patches):
      setattr(obj, name, orig)
  for k, v in caught.items():
    if len(v) == 1:
      saved[f'ours/{k}'] = v[0]
    else:
      for i, x in enumerate(v):
        saved[f'ours/{k}_{i}'] = x
  np.savez(args.out, **saved)
  print(f'wrote {len(saved)} arrays to {args.out}: ' + ' '.join(f'{k}x{len(v)}' for k, v in caught.items()))


def fold(args):
  """A full ColabFold-equivalent fold (optionally with native inputs injected), scored against native's samples."""
  import json
  import jax
  from pathlib import Path
  from alphafold3.common import folding_input
  from alphafold3.model import post_processing
  from colabfold.alphafold3 import models as cf_models

  fold_input = folding_input.Input.from_json(open(args.json).read())
  fold_input = folding_input.Input(name=fold_input.name, chains=fold_input.chains, rng_seeds=[args.seed],
                                   bonded_atom_pairs=fold_input.bonded_atom_pairs, user_ccd=fold_input.user_ccd)
  ex = cf_models.featurise(fold_input, 'chai1', args.model_dir, buckets=None)[0]
  if args.inject:
    _inject(ex, np.load(args.native), args.inject.split(','))
  cfg = cf_models.make_config('chai1', None, 5)
  runner = cf_models.ModelRunner(cfg, Path(args.model_dir))
  runner.__dict__['model_params'] = _patch_params(runner.model_params, args.chain_entity, args.token_dx)
  result = runner.run_inference(ex, jax.random.PRNGKey(args.seed))
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  paths = []
  for i, r in enumerate(runner.extract_inference_results(batch=ex, result=result, target_name=fold_input.name)):
    p = out / f'sample_{i}.cif'
    p.write_bytes(post_processing.post_process_inference_result(r).cif)
    paths.append(p)
  try:
    score(paths, args.native_result)
  except ImportError as e:  # the gauntlet scorer needs gemmi
    print(f'wrote {len(paths)} samples to {out}; score them with a python that has gemmi ({e})')


def score(paths, native_result):
  """Nearest-native polymer RMSD / lDDT and pLDDT, with the gauntlet's own scoring (compare/compare.py)."""
  import importlib.util
  import json
  spec = importlib.util.spec_from_file_location('gauntlet_compare', os.environ.get(
      'GAUNTLET_COMPARE', '/data/milot/uplift/gauntlet/compare/compare.py'))
  cmp = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(cmp)
  nat = [cmp.load(s['structure']) for s in json.load(open(native_result))['samples']]
  ours = [cmp.load(p) for p in paths]
  spread = [cmp.nearest(x, nat, skip=i)['rmsd'] for i, x in enumerate(nat)]
  near = [cmp.nearest(x, nat) for x in ours]
  print(f'native: pLDDT {np.median([x[2] for x in nat]):.1f}  spread median {np.median(spread):.2f} max {max(spread):.2f}')
  print(f'ours:   pLDDT {np.median([x[2] for x in ours]):.1f}  nearest-native RMSD median {np.median([n["rmsd"] for n in near]):.2f} '
        f'(min {min(n["rmsd"] for n in near):.2f})  lDDT median {np.median([n["lddt"] for n in near]):.3f}')
  # per-residue deviation of each sample from its nearest native, after superposition
  dev = []
  for x, n in zip(ours, near):
    a, b = np.concatenate(x[0]), np.concatenate(nat[n['ref']][0])
    r, t = cmp.kabsch(a, b)
    dev.append(np.linalg.norm(a @ r + t - b, axis=-1))
  dev = np.median(np.asarray(dev), 0)
  bad = np.flatnonzero(dev > 2.0)
  print(f'        per-residue median deviation: median {np.median(dev):.2f}; {bad.size} residues > 2 A'
        + (f' at {_ranges(bad)}' if bad.size else ''))


def _ranges(idx):
  out, start = [], idx[0]
  for a, b in zip(idx[:-1], idx[1:]):
    if b != a + 1:
      out.append(f'{start}-{a}' if a > start else f'{start}')
      start = b
  out.append(f'{start}-{idx[-1]}' if idx[-1] > start else f'{start}')
  return ','.join(out)


# PERM[our class] = chai class (converters.chai1.CHAI1_RESTYPE_PERM); our trailing 32nd MSA class is chai's mask 32
CHAI1_RESTYPE_PERM = (list(range(21)) + [31] + [21, 23, 22, 24] + [26, 28, 27, 29] + [25])
_OURS_TO_CHAI = np.asarray(CHAI1_RESTYPE_PERM + [32])


def _stat(tag, a, b, note=''):
  a = np.asarray(a, np.float64).ravel()
  b = np.asarray(b, np.float64).ravel()
  if a.shape != b.shape:
    print(f'  {tag:34s} SHAPE ours {a.shape} native {b.shape}')
    return None
  d = np.abs(a - b)
  same = a.std() > 0 and b.std() > 0
  corr = np.corrcoef(a, b)[0, 1] if same else float(np.array_equal(a, b))
  rb = np.sqrt((b ** 2).mean())
  print(f'  {tag:34s} corr {corr:.6f}  rms ours/nat {np.sqrt((a ** 2).mean()) / max(rb, 1e-12):.4f}  '
        f'max|d| {d.max():.4g}  mean|d| {d.mean():.4g}  n_diff {(d > 1e-4).sum()}/{d.size} {note}')
  return corr


def _token_feats_ours(o, params):
  """Our chai TOKEN stream (pre-concat), rebuilt in numpy from the batch and the blob."""
  def lin(name, x, bias=True):
    p = params[f'diffuser/{name}']
    y = x @ np.asarray(p['weights'], np.float64)
    return y + np.asarray(p['bias'], np.float64) if bias and 'bias' in p else y
  oh = np.eye(31)[o['batch/aatype']]
  prof = np.concatenate([o['batch/profile'], o['batch/deletion_mean'][:, None]], -1)
  return lin('chai1_token_feature_embedding', oh) + lin('chai1_msa_profile_embedding', prof, False)


def compare(args):
  nat = np.load(args.native)
  o = np.load(args.ours)
  n = int(nat['n_tokens'])
  print(f'== {args.native} vs {args.ours}: {n} tokens')
  ctl = np.load(args.control) if args.control else None

  print('-- residue types')
  rt_n = nat['feat/ResidueType'][0, :n, 0]
  rt_o = _OURS_TO_CHAI[o['batch/aatype']]
  bad = np.flatnonzero(rt_n != rt_o)
  print(f'  ResidueType mismatches {bad.size}/{n}' + (f' e.g. tok {bad[:8]} native {rt_n[bad[:8]]} ours->chai {rt_o[bad[:8]]}' if bad.size else ''))

  print('-- MSA rows (native after chai drop_duplicates / pairing; ours after featurise)')
  mt_n, mm_n = nat['inp/msa_tokens'][0, :, :n], nat['inp/msa_mask'][0, :, :n]
  dm_n = nat['inp/msa_deletion_matrix'][0, :, :n].astype(np.int64)
  mt_o, mm_o = o['batch/msa'].astype(np.int64), o['batch/msa_mask']
  dm_o = o['batch/deletion_matrix'].astype(np.int64)
  live_n, live_o = np.flatnonzero(mm_n.any(1)), np.flatnonzero(mm_o.any(1))
  print(f'  live rows native {live_n.size} ours {live_o.size}; num_alignments ours {o["batch/num_alignments"]}')
  print(f'  deletion counts: native max {dm_n.max()}  ours min {dm_o.min()} max {dm_o.max()}')
  k = min(live_n.size, live_o.size)
  tn, to = mt_n[live_n[:k]], _OURS_TO_CHAI[np.clip(mt_o[live_o[:k]], 0, 31)]
  rows_eq = (tn == to).all(1)
  print(f'  first {k} live rows: identical token rows {rows_eq.sum()}; first differing row {np.flatnonzero(~rows_eq)[:5]}')
  if (~rows_eq).any():
    r = np.flatnonzero(~rows_eq)[0]
    c = np.flatnonzero(tn[r] != to[r])[:10]
    print(f'    row {r}: cols {c} native {tn[r, c]} ours {to[r, c]}')
  sn = {bytes(x) for x in mt_n[live_n].astype(np.uint8)}
  so = {bytes(x) for x in _OURS_TO_CHAI[np.clip(mt_o[live_o], 0, 31)].astype(np.uint8)}
  print(f'  row sets: native {len(sn)} unique, ours {len(so)} unique, shared {len(sn & so)}')
  dd = dm_n[live_n[:k]] != dm_o[live_o[:k]]
  print(f'  deletion matrix over the first {k} live rows: {dd.sum()} cells differ in {dd.any(1).sum()} rows')
  if dd.any():
    r, c = np.argwhere(dd)[0]
    print(f'    first at row {r} col {c}: native {dm_n[live_n[r], c]} ours {dm_o[live_o[r], c]}')
  mk = mm_n[live_n[:k]] != mm_o[live_o[:k]]
  print(f'  msa mask differs in {mk.sum()} cells')

  print('-- MSA stream columns, first rows (native generator outputs vs our create_msa_feat)')
  f = o['ours/msa_feat_0'] if 'ours/msa_feat_0' in o.files else o['ours/msa_feat']
  r = f.shape[0]
  _stat('IsPairedMSA', f[..., 0], nat['feat/IsPairedMSA'][0, :r, :n, 0])
  src_n = nat['feat/MSADataSource'][0, :r, :n, 0]
  src_o = f[..., 1:7].argmax(-1)
  print(f'  MSADataSource mismatches {(src_n != src_o).sum()}/{src_n.size}; native classes {np.unique(src_n)} ours {np.unique(src_o)}')
  _stat('MSADeletionValue', f[..., 7], nat['feat/MSADeletionValue'][0, :r, :n, 0])
  _stat('MSAHasDeletion', f[..., 8], nat['feat/MSAHasDeletion'][0, :r, :n, 0])
  oh_o = _OURS_TO_CHAI[f[..., 9:].argmax(-1)]
  oh_n = nat['feat/MSAOneHot'][0, :r, :n, 0]
  print(f'  MSAOneHot mismatches {(oh_n != oh_o).sum()}/{oh_n.size}')

  print('-- token profile / deletion mean')
  prof_n = nat['feat/MSAProfile'][0, :n]
  prof_o = np.zeros_like(prof_n)
  prof_o[:, _OURS_TO_CHAI[:31]] = o['batch/profile']
  _stat('MSAProfile', prof_o, prof_n)
  _stat('MSADeletionMean', o['batch/deletion_mean'], nat['feat/MSADeletionMean'][0, :n, 0])
  for name in ('IsDistillation', 'TokenBFactor', 'TokenPLDDT', 'ChainIsCropped', 'MissingChainContact'):
    v = nat[f'feat/{name}'][0, :n, 0]
    print(f'  {name}: native values {np.unique(v)}')

  print('-- templates')
  for name in ('TemplateMask', 'TemplateResType', 'TemplateDistogram', 'TemplateUnitVector'):
    v = nat[f'feat/{name}']
    print(f'  {name}: native unique {np.unique(v)[:8]}')
  print(f'  ours template_aatype unique {np.unique(o["batch/template_aatype"])} mask any {o["batch/template_atom_mask"].any()}')

  print('-- atoms (per token: names, element, charge, intra-token ref distances)')
  ati = nat['inp/atom_token_index'][0]
  ex = nat['inp/atom_exists_mask'][0]
  names_n = nat['feat/AtomNameOneHot'][0]
  el_n, ch_n = nat['feat/AtomRefElement'][0, :, 0], nat['feat/AtomRefCharge'][0, :, 0]
  pos_n = nat['feat/AtomRefPos'][0] * 10.0
  rm_o = o['batch/ref_mask']
  n_bad_names = n_bad_el = n_bad_ch = 0
  dist_err = []
  cnt_n = np.bincount(ati[ex], minlength=n)[:n]
  cnt_o = rm_o.sum(1)
  bad_cnt = np.flatnonzero(cnt_n != cnt_o)
  print(f'  atoms native {int(ex.sum())} ours {int(rm_o.sum())}; tokens with a different atom count {bad_cnt.size}'
        + (f' e.g. {bad_cnt[:6]} native {cnt_n[bad_cnt[:6]]} ours {cnt_o[bad_cnt[:6]]}' if bad_cnt.size else ''))
  for t in range(n):
    ia = np.flatnonzero(ex & (ati == t))
    io = np.flatnonzero(rm_o[t])
    nn = [tuple(x) for x in names_n[ia]]
    no = [tuple(x) for x in o['batch/ref_atom_name_chars'][t, io]]
    common = [x for x in nn if x in no]
    if sorted(nn) != sorted(no):
      n_bad_names += 1
    a_idx = [ia[nn.index(x)] for x in common]
    o_idx = [io[no.index(x)] for x in common]
    n_bad_el += int((el_n[a_idx] != o['batch/ref_element'][t, o_idx]).any())
    n_bad_ch += int((ch_n[a_idx] != o['batch/ref_charge'][t, o_idx]).any())
    if len(common) > 1:
      pn, po = pos_n[a_idx], o['batch/ref_pos'][t, o_idx]
      dn = np.linalg.norm(pn[:, None] - pn[None], axis=-1)
      do = np.linalg.norm(po[:, None] - po[None], axis=-1)
      dist_err.append(np.abs(dn - do).max())
  dist_err = np.asarray(dist_err)
  print(f'  tokens with different atom-name sets {n_bad_names}, element mismatch {n_bad_el}, charge mismatch {n_bad_ch}')
  if dist_err.size:
    print(f'  intra-token ref distance |d|: median {np.median(dist_err):.4f} max {dist_err.max():.4f} '
          f'tokens > 0.1 A: {(dist_err > 0.1).sum()}')

  print('-- token embedder and trunk (ours vs native' + (', control: native vs native2' if ctl is not None else '') + ')')
  pairs = [('s_trunk (s_init)', 'ours/s_trunk', 'tie/s_trunk'), ('s_struct', 'ours/s_struct', 'tie/s_struct'),
           ('z_init', 'ours/z_init_0', 'tie/z_init')]
  passes = sorted(int(k.rsplit('_', 1)[1]) for k in o.files if k.startswith('ours/single_'))
  for i in passes:
    pairs.append((f'single pass {i}', f'ours/single_{i}', f'trunk/single_{i}'))
    pairs.append((f'pair pass {i}', f'ours/pair_{i}', f'trunk/pair_{i}'))
  for tag, ko, kn in pairs:
    if ko not in o.files or kn not in nat.files:
      continue
    a, b = o[ko], nat[kn][0]
    c = _stat(tag, a, b)
    if ctl is not None and kn in ctl.files:
      _stat('   control', ctl[kn][0], b)
    if args.per_token and a.ndim == 2:
      ca = np.asarray([np.corrcoef(a[t], b[t])[0, 1] for t in range(n)])
      worst = np.argsort(ca)[:12]
      print(f'     per-token corr: median {np.median(ca):.4f}; worst tokens {worst} {np.round(ca[worst], 3)}')
  return 0


def main(argv=None):
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  n = sub.add_parser('native')
  n.add_argument('--fasta', required=True)
  n.add_argument('--msa-dir')
  n.add_argument('--constraints')
  n.add_argument('--work', required=True, help='chai output directory (emptied)')
  n.add_argument('--out', required=True)
  n.add_argument('--seed', type=int, default=1)
  n.add_argument('--recycles', type=int, default=3)
  n.add_argument('--steps', type=int, default=200)
  n.add_argument('--samples', type=int, default=5)
  n.add_argument('--trunk-only', action='store_true')
  n.add_argument('--all-pairs', action='store_true')
  o = sub.add_parser('ours')
  o.add_argument('--json', required=True)
  o.add_argument('--model-dir', required=True)
  o.add_argument('--out', required=True)
  o.add_argument('--seed', type=int, default=1)
  o.add_argument('--recycles', type=int, default=None)
  o.add_argument('--bf16', default=None)
  o.add_argument('--xla', action='store_true', help='XLA attention (fp32 Pallas kernels can exceed shared memory)')
  o.add_argument('--chain-entity', help='.npy of chai1_chain_entity rows for a blob without them')
  o.add_argument('--token-dx', help='.npy of the chai1_token_dx row for a blob without it')
  o.add_argument('--native', help='native npz, for --inject')
  o.add_argument('--inject', help='comma list of native inputs to substitute: profile,deletion_mean,msa,ref_pos')
  f = sub.add_parser('fold')
  f.add_argument('--json', required=True)
  f.add_argument('--model-dir', required=True)
  f.add_argument('--out-dir', required=True)
  f.add_argument('--native-result', required=True, help="the gauntlet native arm's result.json")
  f.add_argument('--seed', type=int, default=1)
  f.add_argument('--chain-entity', help='.npy of chai1_chain_entity rows for a blob without them')
  f.add_argument('--token-dx', help='.npy of the chai1_token_dx row for a blob without it')
  f.add_argument('--native', help='native npz, for --inject')
  f.add_argument('--inject')
  sc = sub.add_parser('score')
  sc.add_argument('native_result')
  sc.add_argument('cifs', nargs='+')
  c = sub.add_parser('compare')
  c.add_argument('native')
  c.add_argument('ours')
  c.add_argument('--control')
  c.add_argument('--per-token', action='store_true')
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]  # tokamax parses absl flags from argv lazily
  if args.cmd == 'score':
    return score(args.cifs, args.native_result)
  return {'native': native, 'ours': ours, 'compare': compare, 'fold': fold}[args.cmd](args)


if __name__ == '__main__':
  raise SystemExit(main())
