"""The STOCK ESMFold2 half of vendor_native_parity.py (esmfold2, esmfold2_fast,
esmfold2_lm600m, esmfold2_lm300m).

  python esmfold2_upstream_runner.py featurise MODEL CASE.json OUT_DIR
  python esmfold2_upstream_runner.py forward   MODEL OUT_DIR IN.npz OUT.npz

Run in ~/venv_esm341 (esm 3.4.1), isolated, HF_HUB_OFFLINE against the cached
biohub checkpoints. Everything is esm's own, driven the way
`ESMFold2InputBuilder.fold` drives it: `prepare_input` (clean_esmfold2_input ->
prepare_esmfold2_input) then `EsmFold2Model.forward`.

Overrides, all of them random draws or the sampler:
  * `structure_head.sample` returns the injected coordinates (which the
    confidence head then scores) and runs the real diffusion module ONCE on the
    injected (noised coordinates, sigma);
  * `_init_pair_state` -- native's recurrence starts from a trunc-normal draw
    where ours starts from zeros, its mean; the draw is returned for the harness
    to inject (`inject/pair_state0`);
  * lm_dropout 0 and msa_column_mask_rate 0: fold()'s inference defaults (0.3,
    0.1) are fresh random masks, which ours does not draw.
The ESM-C hidden states are computed here, by esm's own tower, and handed to
ours (ESMC_HIDDEN): the language model is its own model with its own gate.

The lm-tier pair (esmfold2_lm600m / lm300m) are Synthyra's releases and run
Synthyra's OWN code: the repo snapshot under ~/esmfold2_variants/synthyra
(FastPLMs, trust_remote_code) in ~/venv_synthyra (its requirements.txt), with
its ESM++ tower. The same overrides apply where the code has them (its
experimental loop starts from zeros: no pair-state draw). Its atom attention
defaults to DENSE (every atom attends to every atom, "as the official model does
without flash-attn"); ours, like the official model WITH flash-attn, uses the
+/-64 window, which is what this runner reproduces (fp32, windowed).
"""
import json
import os
import sys

import numpy as np

# The revision each converter read (esmfold2_dumps._REVISION): biohub/ESMFold2's
# main has since moved to a different release (b1324ddf, nested keys, ESM-C
# bundled) that the port does not read; comparing against it would test a
# model we did not convert.
_HUB = os.path.expanduser('~/.cache/huggingface/hub')
_REPO = {
    'esmfold2': _HUB + '/models--biohub--ESMFold2/snapshots/8fc3ff471022fdce52c77030685eb775de0c00a3',
    'esmfold2_fast': _HUB + '/models--biohub--ESMFold2-Fast/snapshots/c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6',
}


_SYNTHYRA = {
    'esmfold2_lm600m': os.path.expanduser('~/esmfold2_variants/synthyra/ESMFold2-600'),
    'esmfold2_lm300m': os.path.expanduser('~/esmfold2_variants/synthyra/ESMFold2-300'),
}
_LOADED = {}


def _types(model):
  """The input dataclasses: esm's, or the Synthyra model's own (same names)."""
  if model in _SYNTHYRA:
    return _model(model).input_types
  import esm.models.esmfold2 as T
  return T


def _structure_input(j, bonds=None, model='esmfold2'):
  T = _types(model)
  DNAInput, LigandInput, MSA, Modification, ProteinInput, RNAInput, StructurePredictionInput = (
      T.DNAInput, T.LigandInput, T.MSA, T.Modification, T.ProteinInput, T.RNAInput, T.StructurePredictionInput)
  seqs = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    cid = ids if len(ids) > 1 else ids[0]
    # AF3's ptmPosition/basePosition is 1-based, esm's Modification.position
    # zero-indexed (esm.utils.structure.input_builder)
    mods = [Modification(position=m.get('ptmPosition', m.get('basePosition')) - 1,
                         ccd=m.get('ptmType', m.get('modificationType')))
            for m in e.get('modifications', [])] or None
    msa = None
    if kind in ('protein', 'rna') and e.get('unpairedMsa') and len(e['unpairedMsa'].splitlines()) > 2:
      import io
      msa = MSA.from_a3m(io.StringIO(e['unpairedMsa']))
    if kind == 'protein':
      seqs.append(ProteinInput(id=cid, sequence=e['sequence'], modifications=mods, msa=msa))
    elif kind == 'rna':
      import dataclasses
      # Synthyra's RNAInput has no msa field (esm's does)
      kw = {'msa': msa} if 'msa' in {f.name for f in dataclasses.fields(RNAInput)} else {}
      seqs.append(RNAInput(id=cid, sequence=e['sequence'], modifications=mods, **kw))
    elif kind == 'dna':
      seqs.append(DNAInput(id=cid, sequence=e['sequence'], modifications=mods))
    elif kind == 'ligand':
      seqs.append(LigandInput(id=cid, smiles=e.get('smiles'), ccd=e.get('ccdCodes')))
  return StructurePredictionInput(sequences=seqs, covalent_bonds=bonds)


def _features(j, seed=1, model='esmfold2'):
  """esm's own featurisation; a declared bond's atom names become esm's per-residue atom indices."""
  if model in _SYNTHYRA:
    builder = _model(model).input_builder
  else:
    from esm.models.esmfold2 import ESMFold2InputBuilder
    builder = ESMFold2InputBuilder()
  CovalentBond = _types(model).CovalentBond

  def prepare(inp):
    if model not in _SYNTHYRA:
      return builder.prepare_input(inp, seed=seed)
    # Synthyra's fold() path: prepare_model_input enforces the checkpoint's MSA
    # contract (msa_conditioning false: an explicit MSA is refused, MSA features dropped)
    try:
      return builder.prepare_model_input(_model(model), inp, seed=seed)
    except ValueError as e:
      if 'without MSA conditioning' in str(e):
        print('VENDOR_NA: %s' % e, file=sys.stderr)
        sys.exit(3)
      raise
  feats, chains = prepare(_structure_input(j, model=model))
  pairs = j.get('bondedAtomPairs') or ()
  if not pairs:
    return feats, chains
  names = _atom_names(feats)
  tok = feats['atom_to_token'][0].numpy()
  real = feats['atom_attention_mask'][0].numpy().astype(bool)
  asym = feats['asym_id'][0].numpy()
  resi = feats['residue_index'][0].numpy()
  chain_asym = {c.chain_id: c.asym_id for c in chains}
  bonds = []
  for a, b in pairs:
    idx = []
    for chain, res, atom in (a, b):
      # esm numbers a residue's atoms in order among the residue's valid atoms
      # esm numbers residues from 0 (enumerate over the sequence); AF3's are 1-based
      sel = [i for i in np.flatnonzero(real) if asym[tok[i]] == chain_asym[chain] and resi[tok[i]] == res - 1]
      idx.append([names[i] for i in sel].index(atom))
    bonds.append(CovalentBond(chain_id1=a[0], res_idx1=a[1] - 1, atom_idx1=idx[0],
                              chain_id2=b[0], res_idx2=b[1] - 1, atom_idx2=idx[1]))
  return prepare(_structure_input(j, bonds, model=model))


def _atom_names(feats):
  chars = feats['ref_atom_name_chars'][0].numpy()
  return [''.join(chr(int(c) + 32) for c in row if c).strip() for row in chars]


def _layout(feats):
  real = feats['atom_attention_mask'][0].numpy().astype(bool)
  tok = feats['token_attention_mask'][0].numpy().astype(bool)
  names = _atom_names(feats)
  return dict(
      atom_name=np.array([n for n, r in zip(names, real) if r]),
      atom_token=feats['atom_to_token'][0].numpy()[real],
      atom_pad=np.ones(int(real.sum()), bool),
      token_pad=tok[tok],
      atom_slot=np.flatnonzero(real),
      **{'feat.ref_pos': feats['ref_pos'][0].numpy()[real],
         'feat.ref_space_uid': feats['ref_space_uid'][0].numpy()[real],
         'feat.ref_element': np.eye(128, dtype=np.int64)[feats['ref_element'][0].numpy()[real]]})


def _dev():
  import torch
  return 'cuda' if torch.cuda.is_available() else 'cpu'


def _fp32_autocast():
  """esm runs the input embedder, folding trunk and heads under bf16 autocast on
  CUDA. Precision is not the model: the vendor side runs fp32, as every other
  vendor here does (bf16 is graded separately). The fused-trimul bf16 cast is
  only taken with the fused kernel backend, which is off by default."""
  import torch
  orig_amp, orig_top = torch.amp.autocast, torch.autocast

  def off(cls):
    class Off(cls):
      def __init__(self, *a, **kw):
        kw['enabled'] = False
        super().__init__(*a, **kw)
    return Off
  torch.amp.autocast, torch.autocast = off(orig_amp), off(orig_top)


def _fp32_atom_attention(L=None):
  """esm's atom attention is bf16 by construction, not by autocast:
  build_3d_rope returns cos/sin `.to(torch.bfloat16)` and
  SWA3DRoPEAttention.forward casts fp32 q/k/v to bf16 before attending (~1e-3
  relative on every atom). Ours is fp32 there. For the logic comparison the
  vendor runs its own code minus those two casts -- build_3d_rope is wrapped to
  recompute in fp32, and the attention forward is esm's non-flash path (the one
  stock takes without flash-attn) with the q/k/v cast left out.
  ESMFOLD2_VENDOR_BF16_ATTN=1 keeps stock's casts."""
  import os
  if os.environ.get('ESMFOLD2_VENDOR_BF16_ATTN'):
    return
  import torch
  import torch.nn.functional as F
  if L is None:
    from esm.models.esmfold2 import layers as L
  orig_rope = L.build_3d_rope

  def rope_fp32(ref_pos, ref_space_uid, head_dim, n_spatial_per_axis=4, n_uid_pairs=2,
                spatial_base_freq=10000.0, uid_base_freq=10.0):
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    dev = ref_pos.device
    sp = 1.0 / (spatial_base_freq ** (torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=dev)
                                      / n_spatial_per_axis))
    ui = 1.0 / (uid_base_freq ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=dev) / n_uid_pairs))
    freqs = torch.cat([torch.einsum('bna,k->bnak', ref_pos.float(), sp).reshape(B, N, 3 * n_spatial_per_axis),
                       torch.einsum('bn,k->bnk', ref_space_uid.float(), ui)], dim=-1)
    if freqs.shape[-1] < half_dim:
      freqs = torch.cat([freqs, torch.zeros(B, N, half_dim - freqs.shape[-1], device=dev)], dim=-1)
    return freqs.cos(), freqs.sin()
  L.build_3d_rope = rope_fp32

  def forward(self, x, attention_params):
    B, N = x.shape[:2]
    cos, sin = attention_params[0], attention_params[1]
    qkv = self.Wqkv(x).view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv.unbind(0)
    q, k = L.qk_norm(q), L.qk_norm(k)
    q, k = L.apply_rotary_emb_3d(q, cos, sin), L.apply_rotary_emb_3d(k, cos, sin)
    if len(attention_params) > 2:
      valid = torch.zeros(B * N, dtype=torch.bool, device=q.device)
      valid[attention_params[2]] = True
      valid = valid.view(B, N)
    else:
      valid = torch.ones(B, N, dtype=torch.bool, device=q.device)
    rank = torch.cumsum(valid, dim=1) - 1
    allowed = ((rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window) \
        & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool, device=q.device)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         attn_mask=allowed.unsqueeze(1), scale=self.scale).transpose(1, 2)
    out = (out * valid.unsqueeze(-1).unsqueeze(-1)).reshape(B, N, -1)
    out = out * torch.sigmoid(self.gate_proj(x))
    return self.out_proj(out)
  L.SWA3DRoPEAttention.forward = forward


def _tower_precision():
  # the stock tower runs bf16; ESM_TOWER_FP32=1 runs it fp32, so the harness can
  # grade OUR tower against it (the fold itself takes these states either way)
  return 'fp32' if os.environ.get('ESM_TOWER_FP32') else 'bf16'


def _model(model_name):
  if model_name in _LOADED:
    return _LOADED[model_name]
  import torch
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  if model_name in _SYNTHYRA:
    # as Synthyra's README loads it (fp32 folding model, bf16 ESM++ tower)
    from transformers import AutoModel
    m = AutoModel.from_pretrained(_SYNTHYRA[model_name], trust_remote_code=True, dtype=torch.float32,
                                  device_map=_dev(), esmc_precision=_tower_precision(),
                                  attn_implementation='sdpa').eval()
  else:
    from esm.models.esmfold2 import EsmFold2Model
    m = EsmFold2Model.from_pretrained(_REPO[model_name], esmc_precision=_tower_precision()).to(_dev()).eval()
  _LOADED[model_name] = m
  return m


def _attention_module(m):
  """The module that defines the model's SWA3DRoPEAttention (esm's layers, or FastPLMs' common)."""
  for mod in m.modules():
    if type(mod).__name__ == 'SWA3DRoPEAttention':
      return sys.modules[type(mod).__module__]
  return None


def featurise(model, case_json, out_dir):
  import torch
  os.makedirs(out_dir, exist_ok=True)
  import shutil
  shutil.copy(case_json, os.path.join(out_dir, 'case.json'))
  with open(case_json) as f:
    j = json.load(f)
  feats, _ = _features(j, model=model)
  np.savez(os.path.join(out_dir, 'layout.npz'), **_layout(feats))
  m = _model(model)
  with torch.no_grad():
    f = {k: v.to(_dev()) for k, v in feats.items() if isinstance(v, torch.Tensor)}
    h = m._compute_lm_hidden_states(f['input_ids'], f['asym_id'], f['residue_index'], f['mol_type'],
                                    f['token_attention_mask'])
  # (num_tokens, num_layers, width), the shape alphafold3.model.esm.shim reads
  np.savez(os.path.join(out_dir, 'esmc_hidden.npz'), lm_hidden=h[0].float().cpu().numpy())


def forward(model, out_dir, in_npz, out_npz):
  import torch
  z = np.load(in_npz)
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  slot = lay['atom_slot']
  with open(os.path.join(out_dir, 'case.json')) as f:
    j = json.load(f)
  feats, _ = _features(j, model=model)
  now = feats['ref_pos'][0].numpy()[slot]
  if not np.array_equal(now, lay['feat.ref_pos']):
    raise RuntimeError('esm re-drew a different reference conformer (max |d| %.3g)'
                       % np.abs(now - lay['feat.ref_pos']).max())
  # precision 'bf16' asks for the vendor's STOCK precision (bf16 grading): its
  # autocast and the atom attention's bf16 casts stay as shipped
  stock = str(z['precision']) == 'bf16'
  if not stock:
    _fp32_autocast()
  m = _model(model)
  if not stock:
    _fp32_atom_attention(_attention_module(m))
  # The checkpoint's own LM dropout (lm_encoder.lm_dropout 0.25, per_loop) is
  # applied by _run_one_loop with training=True in ANY forward, not only under
  # fold(): a fresh random mask every loop. Ours turns it off for the harness,
  # so the vendor does too.
  lm_cfg = getattr(m.config, 'lm_encoder', None)
  if lm_cfg is not None and not isinstance(lm_cfg, dict):
    lm_cfg.lm_dropout = 0.0
  if hasattr(m.config, 'force_lm_dropout_during_inference'):
    # the experimental line's top-level switch (0.0 / False in Synthyra's config)
    m.config.lm_dropout = 0.0
    m.config.force_lm_dropout_during_inference = False
  got = {}
  n_atoms = feats['ref_pos'].shape[1]

  def pad(x):
    out = np.zeros((n_atoms, 3), np.float32)
    out[slot] = x
    return torch.tensor(out, device=_dev())[None]

  if hasattr(m, '_init_pair_state'):
    orig_init = m._init_pair_state

    def init_pair_state(ref):
      s = orig_init(ref)
      got['pair_state0'] = s[0].float().cpu().numpy()
      return s
    m._init_pair_state = init_pair_state
  if os.environ.get('ESM_VENDOR_TAPS'):
    # every top-level module's output, per call, for a stage-by-stage diff
    calls = {}

    def hook(name):
      def fn(_m, _i, o):
        k = calls[name] = calls.get(name, -1) + 1
        for j, x in enumerate(o if isinstance(o, (tuple, list)) else (o,)):
          if torch.is_tensor(x):
            got['tap.%s.%d.%d' % (name, k, j)] = x.detach().float().cpu().numpy()
      return fn
    for name, mod in m.named_children():
      if name not in ('esmc', 'structure_head', 'confidence_head'):
        mod.register_forward_hook(hook(name))
    for name, mod in m.inputs_embedder.named_modules():
      if name:
        mod.register_forward_hook(hook('ie.' + name))
  sh = m.structure_head

  def sample(**kw):
    dm = sh.diffusion_module(
        x_noisy=pad(z['positions_noisy']), t_hat=torch.full((1,), float(z['noise_level']), device=_dev()),
        **{k: kw[k] for k in ('ref_pos', 'ref_charge', 'ref_mask', 'ref_element', 'ref_atom_name_chars',
                              'ref_space_uid', 'tok_idx', 's_inputs', 's_trunk', 'z_trunk',
                              'relative_position_encoding', 'asym_id', 'residue_index', 'entity_id',
                              'token_index', 'sym_id', 'token_attention_mask', 'num_diffusion_samples')},
        return_token_repr=True, return_atom_repr=False, inference_cache=None)
    got['denoised'] = dm['x_denoised'][0].float().cpu().numpy()
    got['z'] = kw['z_trunk'][0].float().cpu().numpy()
    got['s_inputs'] = kw['s_inputs'][0].float().cpu().numpy()
    return {'sample_atom_coords': pad(z['positions']), 'diff_token_repr': dm['token_repr']}
  sh.sample = sample
  hidden = np.load(os.path.join(out_dir, 'esmc_hidden.npz'))['lm_hidden']
  f = {k: v.to(_dev()) for k, v in feats.items() if isinstance(v, torch.Tensor)}
  tok = feats['token_attention_mask'][0].numpy().astype(bool)
  with torch.no_grad():
    h = torch.tensor(hidden, device=_dev())[None]
    import inspect
    accepted = inspect.signature(m.forward).parameters
    # MSA_ALL_ROWS: no random row subsample (max_depth above any depth here)
    extra = {k: v for k, v in dict(lm_mask_pct=0.0, msa_column_mask_rate=0.0,
                                   msa_max_depth=10 ** 6 if os.environ.get('MSA_ALL_ROWS') else None).items()
             if k in accepted and v is not None}
    out = m(**{k: v for k, v in f.items() if k in accepted}, lm_hidden_states=h,
            num_loops=int(z['recycling_steps']), num_diffusion_samples=1, **extra)
  n = int(tok.sum())
  res = dict(s_inputs=got['s_inputs'][:n], z=got['z'][:n, :n],
             distogram_logits=out['distogram_logits'][0].float().cpu().numpy()[:n, :n],
             pae=out['pae'][0].float().cpu().numpy()[:n, :n],
             plddt=out['plddt_per_atom'][0].float().cpu().numpy()[slot] if 'plddt_per_atom' in out
             else out['plddt'][0].float().cpu().numpy(),
             denoised=got['denoised'][slot],
             **{k: v for k, v in got.items() if k.startswith('tap.')})
  if 'pde' in out:      # the lm-tier releases have no PDE head
    res['pde'] = out['pde'][0].float().cpu().numpy()[:n, :n]
  if 'pair_state0' in got:
    res['pair_state0'] = got['pair_state0'][:n, :n]
  np.savez(out_npz, **res)


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward}[cmd](*rest)
