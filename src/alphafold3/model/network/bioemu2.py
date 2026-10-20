"""BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) in JAX.

BioEmu2 is Boltz-derived but shares neither of AlphaFold 3's two big stages:

  * the trunk is a PairMixer: Boltz's MSA module with triangle ATTENTION removed,
    then 48 blocks of triangle multiplication (out, in) and a transition. The
    single representation is never updated inside a pass, only re-projected
    from the previous pass by the recycle;
  * the denoiser is a "dual-stream" stack: 16 layers that each run a windowed
    atom transformer and a full token transformer side by side (DiT adaLN-Zero
    blocks) and then mix the two streams through every token's centre atom.

So it is a module of its own rather than more branches on the shared graph.
Everything here is a transcription of bioemu2/biomoa (@7846769), and the
parameters keep the vendor's own names and layouts: one Haiku scope,
`diffuser/~/bioemu2`, whose parameter names are the PyTorch state-dict keys
with '.' -> '/' (Linear weights stay (out, in)). The 48 PairMixer, 4 MSA and 16
dual-stream layers are STACKED on a leading axis (`.../layers/...`) and run
under lax.scan. A numbered nn.Sequential / ModuleList entry anywhere else
becomes `name_<i>`. converters/bioemu2.py is therefore a rename.

The network reads BioEmu2's own feature dict (the vendor's batch keys, no batch
axis): tokens T, atoms N padded to a multiple of 32, MSA rows S. Building that
dict from an AlphaFold 3 batch is the featuriser's job, graded separately.
"""

import math

import haiku as hk
import jax
import jax.numpy as jnp

W_Q, W_K = 32, 128          # atom query / key window; keys are centred on queries
CTX = (W_K - W_Q) // 2      # 48 key atoms of context on each side
INF = 1e6
SIGMA_DATA = 16.0
NUM_TOKENS = 33
NUM_POCKET = 4


# --- primitives ---------------------------------------------------------------

def _ln(x, w=None, b=None, eps=1e-5):
  """torch.nn.functional.layer_norm over the last axis (biased variance)."""
  x = x.astype(jnp.float32)
  mu = x.mean(-1, keepdims=True)
  var = jnp.square(x - mu).mean(-1, keepdims=True)
  y = (x - mu) * jax.lax.rsqrt(var + eps)
  if w is not None:
    y = y * w
  if b is not None:
    y = y + b
  return y


def _lin(x, w, b=None):
  """nn.Linear with a torch-layout (out, in) weight."""
  y = jnp.einsum('...i,oi->...o', x, w)
  return y if b is None else y + b


def _transition(p, x, norm=True):
  """SwiGLU Transition: fc3(silu(fc1(LN x)) * fc2(LN x))."""
  if norm:
    x = _ln(x, p['norm/weight'], p['norm/bias'])
  return _lin(jax.nn.silu(_lin(x, p['fc1/weight'])) * _lin(x, p['fc2/weight']),
              p['fc3/weight'])


def _ada_ln(x, alpha, beta):
  return _ln(x) * (1 + alpha) + beta


def _fourier(p, t):
  """FourierEmbedding: cos(2 pi (w t + b)), a frozen random projection."""
  return jnp.cos(2 * math.pi * (t[..., None] * p['proj/weight'][:, 0] + p['proj/bias']))


def _sub(p, prefix):
  """The entries of a flat param dict under `prefix/`, with the prefix stripped."""
  n = len(prefix) + 1
  return {k[n:]: v for k, v in p.items() if k.startswith(prefix + '/')}


# --- atom <-> token -----------------------------------------------------------

def _tokens_to_atoms(x_tok, atom_tok, atom_pad):
  """broadcast_tokens_to_atoms: gather, zero on padding atoms."""
  return jnp.where(atom_pad[:, None], x_tok[atom_tok], 0.0)


def _centre_atoms_to_tokens(x_atom, centre, token_pad):
  """gather_center_tokens_from_atomic: each token's centre atom, zero on padding tokens."""
  return jnp.where(token_pad[:, None], x_atom[centre], 0.0)


def _mean_atoms_to_tokens(x_atom, atom_tok, atom_pad, num_tokens):
  """scatter_mean_atoms_to_tokens: per-token mean over real atoms (+1e-6 in the count)."""
  w = atom_pad.astype(jnp.float32)
  s = jax.ops.segment_sum(x_atom * w[:, None], atom_tok, num_segments=num_tokens)
  n = jax.ops.segment_sum(w, atom_tok, num_segments=num_tokens)
  return s / (n[:, None] + 1e-6)


def _key_windows(x, fill=0):
  """Key side of the centred windows: keys[k, j] = x[k*W_Q - CTX + j], out-of-range -> fill.

  pad + unfold(size=W_K, step=W_Q), as the vendor gathers every key window."""
  n = x.shape[0]
  nw = n // W_Q
  pad = jnp.pad(x, [(CTX, CTX)] + [(0, 0)] * (x.ndim - 1), constant_values=fill)
  idx = jnp.arange(nw)[:, None] * W_Q + jnp.arange(W_K)[None, :]
  return pad[idx]                                               # (NW, W_K, ...)


def _query_windows(x):
  return x.reshape((x.shape[0] // W_Q, W_Q) + x.shape[1:])     # (NW, W_Q, ...)


# --- attention ----------------------------------------------------------------

def _attention(p, x, bias, key_mask, heads):
  """AttentionPairBias, full: x (T, D), bias (H, T, T) already projected, key_mask (T,)."""
  t, d = x.shape
  hd = d // heads
  q, k, v = jnp.split(_lin(x, p['proj_qkv/weight']), 3, axis=-1)
  q = (q + p['proj_q_bias']).reshape(t, heads, hd)
  k = k.reshape(t, heads, hd)
  v = v.reshape(t, heads, hd)
  g = jax.nn.sigmoid(_lin(x, p['proj_g/weight']))
  logits = jnp.einsum('ihd,jhd->hij', q, k) / hd ** 0.5 + bias
  logits = jnp.where(key_mask[None, None, :], logits, -INF)
  o = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, axis=-1), v).reshape(t, d)
  return _lin(g * o, p['proj_o/weight'])


def _pair_bias(p, z):
  """LN(z) (no offset) -> per-head bias, heads first."""
  b = _lin(_ln(z, p['norm_z/weight']), p['proj_z/weight'])
  return jnp.moveaxis(b, -1, -3)


def _window_attention(p, x, pair, atom_pad, heads):
  """AttentionPairBiasSlidingWindow: x (N, D), pair (NW, W_Q, W_K, c_z), atom_pad (N,)."""
  n, d = x.shape
  hd = d // heads
  qkv = _lin(x, p['proj_qkv/weight'])
  q_raw, kv_raw = qkv[:, :d], qkv[:, d:]
  q = _query_windows((q_raw + p['proj_q_bias']).reshape(n, heads, hd))     # (NW, W, H, hd)
  kv = _key_windows(kv_raw)                                                 # (NW, K, 2D)
  # the vendor's fused k/v layout: reshape(..., 2, heads, hd)
  kv = kv.reshape(kv.shape[0], W_K, 2, heads, hd)
  k, v = kv[:, :, 0], kv[:, :, 1]
  g = jax.nn.sigmoid(_lin(x, p['proj_g/weight']))
  bias = jnp.moveaxis(_lin(_ln(pair, p['norm_z/weight']), p['proj_z/weight']), -1, 1)
  logits = jnp.einsum('wihd,wjhd->whij', q, k) / hd ** 0.5 + bias          # (NW, H, W, K)
  key_mask = _key_windows(atom_pad, fill=False)                             # (NW, K)
  logits = jnp.where(key_mask[:, None, None, :], logits, -INF)
  o = jnp.einsum('whij,wjhd->wihd', jax.nn.softmax(logits, axis=-1), v).reshape(n, d)
  return _lin(g * o, p['proj_o/weight'])


def _conditioned_layer(p, x, pair, c, mask, heads, windowed, lns=1.0):
  """ConditionedTransformerLayer: x + gamma * op(lns * ((1 + alpha) LN x + beta)), twice."""
  cond = _lin(jax.nn.silu(_ln(c, p['cond_0/weight'], p['cond_0/bias'])),
              p['cond_2/weight'], p['cond_2/bias'])
  a_al, a_be, a_ga, m_al, m_be, m_ga = jnp.split(cond, 6, axis=-1)
  h = lns * _ada_ln(x, a_al, a_be)
  if windowed:
    h = _window_attention(_sub(p, 'attention/attention_layer/pair_bias_attn'),
                          h, pair, mask, heads)
  else:
    h = _attention(_sub(p, 'attention/pair_bias_attn'), h,
                   _pair_bias(_sub(p, 'attention/pair_bias_attn'), pair), mask, heads)
  x = x + a_ga * h
  h = _transition(_sub(p, 'transition'), lns * _ada_ln(x, m_al, m_be), norm=False)
  return x + m_ga * h


# --- trunk ----------------------------------------------------------------------

def _trimul(p, z, pair_mask, outgoing):
  """Boltz-2 triangle multiplication (TriangleMultiplicationFull)."""
  zn = _ln(z, p['norm_in/weight'], p['norm_in/bias'])
  proj = _lin(zn, p['p_in/weight']) * jax.nn.sigmoid(_lin(zn, p['g_in/weight']))
  proj = jnp.where(pair_mask[..., None], proj, 0.0)
  a, b = jnp.split(proj, 2, axis=-1)
  if outgoing:
    out = jnp.einsum('ikd,jkd->ijd', a, b)
  else:
    out = jnp.einsum('kid,kjd->ijd', a, b)
  return (_lin(_ln(out, p['norm_out/weight'], p['norm_out/bias']), p['p_out/weight'])
          * jax.nn.sigmoid(_lin(zn, p['g_out/weight'])))


def _pair_weighted_averaging(p, m, z, pair_mask, heads=8, c_h=32):
  s, t, _ = m.shape
  m = _ln(m, p['norm_m/weight'], p['norm_m/bias'])
  v = _lin(m, p['proj_m/weight']).reshape(s, t, heads, c_h)
  b = jnp.moveaxis(_lin(_ln(z, p['norm_z/weight'], p['norm_z/bias']), p['proj_z/weight']), -1, 0)
  b = jnp.where(pair_mask[None], b, -INF)
  w = jax.nn.softmax(b, axis=-1)                                   # (H, T, T)
  g = jax.nn.sigmoid(_lin(m, p['proj_g/weight']))
  o = jnp.einsum('hij,sjhd->sihd', w, v).reshape(s, t, heads * c_h)
  return _lin(g * o, p['proj_o/weight'])


def _outer_product_mean(p, m, msa_mask):
  """OuterProductMeanVanilla: the mean is over VALID ROWS, not per pair."""
  proj = _lin(_ln(m, p['norm/weight'], p['norm/bias']), p['proj_ab/weight'])
  proj = jnp.where(msa_mask[..., None], proj, 0.0)
  a, b = jnp.split(proj, 2, axis=-1)
  n_valid = jnp.maximum(msa_mask.any(axis=-1).sum(), 1).astype(jnp.float32)
  o = jnp.einsum('sic,sjd->ijcd', a, b)
  o = o.reshape(o.shape[:2] + (-1,)) / n_valid
  return _lin(o, p['proj_o/weight'], p['proj_o/bias'])


def _msa_layer(p, z, m, pair_mask, msa_mask):
  m = m + _pair_weighted_averaging(_sub(p, 'pair_weighted_averaging'), m, z, pair_mask)
  m = m + _transition(_sub(p, 'msa_transition'), m)
  z = z + _outer_product_mean(_sub(p, 'outer_product_mean'), m, msa_mask)
  z = z + _trimul(_sub(p, 'tri_mul_out'), z, pair_mask, True)
  z = z + _trimul(_sub(p, 'tri_mul_in'), z, pair_mask, False)
  z = z + _transition(_sub(p, 'z_transition'), z)
  return z, m


def _pairmixer_layer(p, z, pair_mask):
  z = z + _trimul(_sub(p, 'tri_mul_out'), z, pair_mask, True)
  z = z + _trimul(_sub(p, 'tri_mul_in'), z, pair_mask, False)
  return z + _transition(_sub(p, 'transition_z'), z)


def _relpos(p, f, r_max=32, s_max=2):
  """FusedRelativePositionEncoder."""
  asym, ent, res, sym = f['asym_id'], f['entity_id'], f['residue_index'], f['sym_id']
  tok = jnp.arange(asym.shape[0])
  same_chain = asym[:, None] == asym[None, :]
  same_res = res[:, None] == res[None, :]
  same_ent = ent[:, None] == ent[None, :]
  d_res = jnp.clip(res[:, None] - res[None, :] + r_max, 0, 2 * r_max)
  d_res = jnp.where(same_chain, d_res, 2 * r_max + 1)
  d_tok = jnp.clip(tok[:, None] - tok[None, :] + r_max, 0, 2 * r_max)
  d_tok = jnp.where(same_chain & same_res, d_tok, 2 * r_max + 1)
  d_chain = jnp.clip(sym[:, None] - sym[None, :] + s_max, 0, 2 * s_max)
  d_chain = jnp.where(same_ent, d_chain, 2 * s_max + 1)
  return (p['embed_rel_pos/weight'][d_res] + p['embed_rel_token/weight'][d_tok]
          + same_ent[..., None].astype(jnp.float32) * p['embed_same_entity/weight'][:, 0]
          + p['embed_rel_chain/weight'][d_chain])


def _contact_conditioning(p, f, pair_mask, cutoff_min=4.0, cutoff_max=20.0):
  cc = f['contact_conditioning'].astype(jnp.float32)
  thr = (f['contact_threshold'].astype(jnp.float32) - cutoff_min) / (cutoff_max - cutoff_min)
  x = jnp.concatenate([cc[..., 2:], thr[..., None], _fourier(_sub(p, 'fourier_embedding'), thr)], -1)
  x = _lin(x, p['encoder/weight'], p['encoder/bias'])
  x = (x * (1 - cc[..., 0:2].sum(-1, keepdims=True))
       + p['encoding_unspecified'] * cc[..., 0:1]
       + p['encoding_unselected'] * cc[..., 1:2])
  return jnp.where(pair_mask[..., None], x, 0.0)


# --- atoms ----------------------------------------------------------------------

def _atom_features(p, f):
  """The shared front of AtomAttentionEncoder.compute_static_context: c and the
  ref-geometry half of p. Returns (c (N, atom_s), p (NW, W_Q, W_K, atom_z))."""
  pad = f['atom_pad_mask']
  ref = f['template_coords'].astype(jnp.float32)
  c = _lin(jnp.concatenate([ref, f['charge'][:, None].astype(jnp.float32)], -1),
           p['proj_atom_continuous/weight'])
  c = c + p['embed_element/weight'][f['element']]
  for i in range(4):
    c = c + p[f'embed_atom_name_chars_{i}/weight'][f['name_encoding'][:, i]]
  c = jnp.where(pad[:, None], c, 0.0)
  # windowed_atom_pair_geometry
  disp = _key_windows(ref)[:, None, :, :] - _query_windows(ref)[:, :, None, :]   # (NW, W, K, 3)
  inv = 1.0 / (1.0 + jnp.sum(disp * disp, -1, keepdims=True))
  uid = jnp.where(pad, f['ref_space_uid'], -1)
  same = (_query_windows(pad)[:, :, None]
          & (_query_windows(f['ref_space_uid'])[:, :, None] == _key_windows(uid, fill=-1)[:, None, :]))
  same = same[..., None].astype(jnp.float32)
  pr = (_lin(disp, p['embed_atompair_ref_pos/weight'])
        + _lin(inv, p['embed_atompair_ref_dist/weight'])
        + _lin(same, p['embed_atompair_mask/weight']))
  return c, pr * same


def _atom_pair_tail(p, c, pr):
  """c -> p (queries and keys), then the p MLP."""
  pr = pr + _lin(jax.nn.relu(_query_windows(c)[:, :, None, :]), p['c_to_p_trans_q_1/weight'])
  pr = pr + _lin(jax.nn.relu(_key_windows(c)[:, None, :, :]), p['c_to_p_trans_k_1/weight'])
  h = _lin(jax.nn.relu(pr), p['p_mlp_1/weight'])
  h = _lin(jax.nn.relu(h), p['p_mlp_3/weight'])
  h = _lin(jax.nn.relu(h), p['p_mlp_5/weight'])
  return pr + h


def _token_pair_to_windows(z, atom_tok, atom_pad):
  """broadcast_token_pair_to_atom_window: z[tok_q, tok_k], zero where either atom is padding."""
  tq = _query_windows(atom_tok)
  tk = _key_windows(atom_tok, fill=0)
  pq = _query_windows(atom_pad)
  pk = _key_windows(atom_pad, fill=False)
  out = z[tq[:, :, None], tk[:, None, :]]
  return jnp.where((pq[:, :, None] & pk[:, None, :])[..., None], out, 0.0)


# --- time ---------------------------------------------------------------------------

def _damped_harmonic(dt, taus, q):
  """DampedHarmonicTimeEmbedding for a scalar lag: [envelope, sin, cos], (n * (1 + 2|q|),).

  taus and the quality factors are the checkpoint's own buffers, not recomputed:
  a float32 logspace in JAX need not round as torch's does."""
  x = dt / taus
  env = jnp.exp(-x)
  phase = 2 * math.pi * q[:, None] * x[None, :]                       # (Q, n)
  # the vendor's (B, Q, T) -> transpose(1, 2) -> (B, T*Q) order: timescale-major
  sin = (env[None] * jnp.sin(phase)).T.reshape(-1)
  cos = (env[None] * jnp.cos(phase)).T.reshape(-1)
  return jnp.concatenate([env, sin, cos])


# --- the module ---------------------------------------------------------------------

class BioEmu2(hk.Module):
  """All of BioEmu2's parameters, under one scope; see the module docstring."""

  def __init__(self, name='bioemu2'):
    super().__init__(name=name)

  def _p(self, prefix, shapes):
    return {k: hk.get_parameter(prefix + '/' + k, s, init=hk.initializers.RandomNormal(0.02))
            for k, s in shapes.items()}

  def _params(self, prefix):
    """Every parameter under `prefix`, by the shapes in _SHAPES."""
    return self._p(prefix, {k[len(prefix) + 1:]: s for k, s in _SHAPES.items()
                            if k.startswith(prefix + '/')})

  # -- trunk ------------------------------------------------------------------------

  def input_embedder(self, f):
    """InputEmbedder -> s_inputs (T, 455)."""
    p = self._params('input_embedder/atom_attention_encoder')
    pad = f['atom_pad_mask']
    c, pr = _atom_features(p, f)
    pr = _atom_pair_tail(p, c, pr)
    q = c
    for i in range(3):
      q = _conditioned_layer(_sub(p, f'atom_encoder_layers_{i}/transformer_layer'),
                             q, pr, c, pad, heads=4, windowed=True)
    num_tokens = f['res_type'].shape[0]
    a = _mean_atoms_to_tokens(jax.nn.relu(_lin(q, p['atom_to_token_trans_0/weight'])),
                              f['token_idx_at_atom'], pad, num_tokens)
    return jnp.concatenate([
        a, jax.nn.one_hot(f['res_type'], NUM_TOKENS), f['profile'].astype(jnp.float32),
        f['deletion_mean'][:, None].astype(jnp.float32),
        jax.nn.one_hot(f['pocket_feature'], NUM_POCKET)], -1)

  def trunk_init(self, f, s_inputs):
    """s_init and z_init: everything before the recycle loop."""
    p = self._params('sequence_embedder')
    tpad = f['token_pad_mask']
    pair_mask = tpad[:, None] & tpad[None, :]
    s_init = _lin(s_inputs, p['s_init/weight'])
    z_init = (_lin(s_inputs, p['z_init_1/weight'])[:, None]
              + _lin(s_inputs, p['z_init_2/weight'])[None, :])
    meth = f['method_feature']
    s_init = s_init + jnp.where(tpad[:, None], p['method_conditioning/embedding/weight'][meth], 0.0)
    pe = p['method_conditioning/pair_embedding/weight'][meth]
    z_init = z_init + jnp.where(pair_mask[..., None], pe[:, None] + pe[None, :], 0.0)
    z_init = z_init + _relpos(_sub(p, 'rel_pos'), f)
    z_init = z_init + _lin(f['token_bonds'].astype(jnp.float32), p['token_bonds/weight'])
    z_init = z_init + _contact_conditioning(_sub(p, 'contact_conditioning'), f, pair_mask)
    return s_init, z_init

  def trunk_pass(self, f, s_inputs, s_init, z_init, s_prev, z_prev):
    """One pass of the recycle loop."""
    p = self._params('sequence_embedder')
    tpad = f['token_pad_mask']
    pair_mask = tpad[:, None] & tpad[None, :]
    s = s_init + _lin(_ln(s_prev, p['s_norm/weight'], p['s_norm/bias']), p['s_recycle/weight'])
    z = z_init + _lin(_ln(z_prev, p['z_norm/weight'], p['z_norm/bias']), p['z_recycle/weight'])
    # MSA module
    msa_feat = jnp.concatenate([
        f['msa'].astype(jnp.float32), f['has_deletion'][..., None].astype(jnp.float32),
        f['deletion_value'][..., None].astype(jnp.float32),
        f['msa_paired'][..., None].astype(jnp.float32)], -1)
    m = _lin(msa_feat, p['msa_module/msa_proj/weight']) + _lin(s_inputs, p['msa_module/s_proj/weight'])[None]
    msa_mask = f['msa_mask'].astype(bool)
    lp = _sub(p, 'msa_module/layers')

    def msa_body(carry, lp_i):
      z, m = carry
      return _msa_layer(lp_i, z, m, pair_mask, msa_mask), None
    (z, m), _ = jax.lax.scan(msa_body, (z, m), lp)
    lp = _sub(p, 'pairmixer_module/layers')

    def mixer_body(z, lp_i):
      return _pairmixer_layer(lp_i, z, pair_mask), None
    z, _ = jax.lax.scan(mixer_body, z, lp)
    return s, z

  def distogram(self, z):
    p = self._params('distogram_module')
    return _lin(z + jnp.swapaxes(z, 0, 1), p['distogram/weight'], p['distogram/bias'])

  # -- denoiser ---------------------------------------------------------------------

  def static_context(self, f, s_trunk, z_trunk):
    """DualStreamDiffusionModule.compute_static_context: everything that does not
    depend on the coordinates, computed once per fold."""
    p = self._params('structure_module')
    pad = f['atom_pad_mask']
    relpos = _relpos(self._params('rel_pos'), f)
    pc = _sub(p, 'pairwise_conditioner')
    z = _lin(_ln(jnp.concatenate([z_trunk, relpos], -1),
                 pc['dim_pairwise_init_proj_0/weight'], pc['dim_pairwise_init_proj_0/bias']),
             pc['dim_pairwise_init_proj_1/weight'])
    for i in range(2):
      z = z + _transition(_sub(pc, f'transitions_{i}'), z)
    pa = _sub(p, 'atom_feature_embedder')
    c, pr = _atom_features(pa, f)
    q0 = c
    s_to_c = _lin(_ln(s_trunk, pa['s_to_c_trans_0/weight'], pa['s_to_c_trans_0/bias']),
                  pa['s_to_c_trans_1/weight'])
    c = c + _tokens_to_atoms(s_to_c, f['token_idx_at_atom'], pad)
    z_to_p = _lin(_ln(z_trunk, pa['z_to_p_trans_0/weight'], pa['z_to_p_trans_0/bias']),
                  pa['z_to_p_trans_1/weight'])
    pr = pr + _token_pair_to_windows(z_to_p, f['token_idx_at_atom'], pad)
    pr = _atom_pair_tail(pa, c, pr)
    return {'z': z, 'q0': q0, 'p': pr}

  def denoise(self, f, ctx, s_inputs, s_trunk, coords, sigma):
    """One denoiser call: noisy coords (N, 3) in angstrom at noise level sigma ->
    denoised coords (N, 3). EDM preconditioning as BioMoaCModel.denoise."""
    p = self._params('structure_module')
    pad = f['atom_pad_mask']
    atom_tok = f['token_idx_at_atom']
    centre = f['token_center_atom_index']
    tpad = f['token_pad_mask']
    num_tokens = tpad.shape[0]
    c_in = 1.0 / jnp.sqrt(sigma ** 2 + SIGMA_DATA ** 2)
    r = jnp.where(f['atom_mask'][:, None], coords * c_in, 0.0)
    t = 0.25 * jnp.log(sigma / SIGMA_DATA)

    # single conditioning
    ps = _sub(p, 'single_conditioner')
    s = _lin(_ln(jnp.concatenate([s_trunk, s_inputs], -1),
                 ps['norm_single/weight'], ps['norm_single/bias']),
             ps['single_embed/weight'], ps['single_embed/bias'])
    four = _ln(_fourier(_sub(ps, 'fourier_embed'), t), ps['norm_fourier/weight'], ps['norm_fourier/bias'])
    s = s + _lin(four, ps['fourier_to_single/weight'])[None]
    is_dyn = f.get('is_dynamics_conditioned', jnp.zeros((), bool))
    dt = f.get('dt', jnp.zeros((), jnp.float32))
    s = s + (_lin(_damped_harmonic(dt, ps['dt_embed/taus'], ps['dt_embed/quality_factors']),
                  ps['dt_embed_to_single/weight'])
             * is_dyn.astype(jnp.float32))[None]
    for i in range(2):
      s = s + _transition(_sub(ps, f'transitions_{i}'), s)
    cond = s

    # the atom stream's initial state
    pa = _sub(p, 'atom_feature_embedder')
    q = ctx['q0'] + _lin(r, pa['r_to_q_trans/weight'])
    if 'coords_cond' in f:
      contrib = _lin(f['coords_cond'], pa['cond_to_q_trans/weight'])
      decay = jnp.exp(-dt / pa['dt_gate_embed/taus'])
      gate = (decay * jax.nn.softplus(pa['dt_gate_log_w'])).sum()
      q = q + jnp.where(is_dyn, contrib * gate, 0.0)

    a = _lin(_ln(s_trunk, p['s_trunk_to_a_0/weight'], p['s_trunk_to_a_0/bias']),
             p['s_trunk_to_a_1/weight'])
    a = a + _lin(_centre_atoms_to_tokens(r, centre, tpad), p['r_token_to_a/weight'])

    c_to_q = _tokens_to_atoms(cond, atom_tok, pad)

    def layer(carry, lp):
      a, q = carry
      q = _conditioned_layer(_sub(lp, 'atom_block/transformer_layer'), q, ctx['p'], c_to_q,
                             pad, heads=4, windowed=True)
      a = _conditioned_layer(_sub(lp, 'token_block'), a, ctx['z'], cond, tpad,
                             heads=16, windowed=False)
      cp = _sub(lp, 'coupling')
      d_a, d_q = a.shape[-1], q.shape[-1]
      al_q, be_q, ga_q = jnp.split(
          _lin(jax.nn.silu(_ln(c_to_q, cp['q_coupling_cond_0/weight'], cp['q_coupling_cond_0/bias'])),
               cp['q_coupling_cond_2/weight'], cp['q_coupling_cond_2/bias']),
          [d_a + d_q, 2 * (d_a + d_q)], axis=-1)
      al_s, be_s, ga_s = jnp.split(
          _lin(jax.nn.silu(_ln(cond, cp['s_coupling_cond_0/weight'], cp['s_coupling_cond_0/bias'])),
               cp['s_coupling_cond_2/weight'], cp['s_coupling_cond_2/bias']),
          [d_a + d_q, 2 * (d_a + d_q)], axis=-1)
      s_to_q = jnp.concatenate([_tokens_to_atoms(a, atom_tok, pad), q], -1)
      s_to_q = ga_q * _transition(_sub(cp, 's_to_q_mixer'), _ada_ln(s_to_q, al_q, be_q), norm=False)
      q_to_s = jnp.concatenate([_centre_atoms_to_tokens(q, centre, tpad), a], -1)
      q_to_s = ga_s * _transition(_sub(cp, 'q_to_s_mixer'), _ada_ln(q_to_s, al_s, be_s), norm=False)
      return (a + q_to_s, q + s_to_q), None
    (a, q), _ = jax.lax.scan(layer, (a, q), _sub(p, 'layers'))

    pr = _sub(p, 'readout')
    h = _lin(q, pr['q_to_hidden/weight']) + _tokens_to_atoms(
        _lin(a, pr['s_to_hidden/weight']), atom_tok, pad)
    out = _lin(_ln(h, pr['to_out_0/weight']), pr['to_out_1/weight'])
    c_skip = SIGMA_DATA ** 2 / (sigma ** 2 + SIGMA_DATA ** 2)
    c_out = sigma * SIGMA_DATA / jnp.sqrt(SIGMA_DATA ** 2 + sigma ** 2)
    del num_tokens
    return c_skip * coords + c_out * out


def _load_shapes():
  """Every parameter of bioemu2-preview: Haiku name (see the module docstring)
  -> shape. Written from the checkpoint by converters/bioemu2.py, which checks
  the blob against it, so the graph and the weights cannot drift apart."""
  import json
  import os
  with open(os.path.join(os.path.dirname(__file__), 'bioemu2_params.json')) as fh:
    return {k: tuple(v) for k, v in json.load(fh).items()}


_SHAPES = _load_shapes()
