"""HelixFold3 (Baidu PaddleHelix) -> AF3 haiku weight converter.

    python -m converters.convert --model helixfold3 --out ~/ported/helixfold3

The checkpoint is a PaddlePaddle `.pdparams`: a pickle of numpy arrays, so it
reads without paddle. Paddle's `nn.Linear.weight` is (in, out) -- haiku's own
layout -- so unlike every torch converter here NOTHING is transposed by default;
only the attention projections are reshaped into heads.

HelixFold3 is an AF3 reproduction: the module tree is AF3's, and the
differences this graph expresses as flags (model_config.*, all naming
helixfold3) are
  * biased Linears in the triangle multiplication (six), triangle attention
    (gate, output) and outer product mean (left/right projections);
  * the relative-position block keyed on an equal residue index;
  * the atom-pair inverse squared offset taken per axis, (3, 16);
  * sequence-index RoPE in the diffusion transformers (interleaved in its code,
    so q/k columns are permuted here into our half-split pairing);
  * protenix's template forward on its own 108 features, an outer residual
    around the template stack, and a same-chain template stack mask;
  * a 32-class residue alphabet (unknown DNA its own class, 26).
Its per-block pair-bias LayerNorms in the diffusion transformers fold into
AF3's shared one: the pair input is the same for every block, so each block's
scale folds into its projection, and the offset adds a per-head constant to
every logit, which the softmax ignores.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import common as C
from .openfold3 import _reorder_features_1d

# AF3 residue class -> HelixFold3 class. Ours: 0-19 amino acids, 20 UNK, 21 gap,
# 22-25 A G C U, 26-29 DA DG DC DT, 30 N. HelixFold3's: 0-21 the same, 22-26
# DA DG DC DT DN, 27-31 A G C U N -- unknown DNA (26) has no class of ours.
AF3_TO_HF3 = np.array(list(range(22)) + [27, 28, 29, 30, 22, 23, 24, 25, 31])
HF3_UNK_DNA = 26

E = 'diffuser/evoformer'
PF = f'{E}/__layer_stack_no_per_layer_1/trunk_pairformer'
MS = f'{E}/__layer_stack_no_per_layer/msa_stack'
TE = f'{E}/template_embedding'
TS = f'{TE}/__layer_stack_no_per_layer/tmpl_pairformer'
DH = 'diffuser/~/diffusion_head'
CH = 'diffuser/confidence_head'
CP = f'{CH}/__layer_stack_no_per_layer/confidence_pairformer'
EP = 'embeddings_and_pairformer'


def load_checkpoint(path):
  """The .pdparams, or the release zip holding it."""
  path = str(path)
  if path.endswith('.zip'):
    import zipfile
    with zipfile.ZipFile(path) as z:
      name = next(n for n in z.namelist()
                  if n.endswith('.pdparams') and not n.split('/')[-1].startswith('._'))
      sd = pickle.loads(z.read(name))
  else:
    with open(path, 'rb') as fh:
      sd = pickle.load(fh)
  return {k: np.asarray(v, np.float32) for k, v in sd.items()
          if isinstance(v, np.ndarray)}


class _Reader:
  """The state dict, recording what was read (audit_coverage reads `seen`)."""

  def __init__(self, sd):
    self.sd = sd

  def __call__(self, key):
    if key not in self.sd:
      raise KeyError(f'helixfold3 converter: missing source key {key!r}')
    return self.sd[key]


def _ln(g, prefix, scale_only=False):
  out = {'scale': g(f'{prefix}.weight')}
  if not scale_only:
    out['offset'] = g(f'{prefix}.bias')
  return out


def _nest(prefix, d):
  return {f'{prefix}/{k}': v for k, v in d.items()}


def _put(params, scope, local):
  """{'sub/param': arr} or {'param': arr} under `scope`."""
  for k, v in local.items():
    if '/' in k:
      sub, name = k.rsplit('/', 1)
      params.setdefault(f'{scope}/{sub}', {})[name] = np.asarray(v, np.float32)
    else:
      params.setdefault(scope, {})[k] = np.asarray(v, np.float32)


def _stack(dicts):
  return {k: np.stack([d[k] for d in dicts]) for k in dicts[0]}


# --- pair block: triangle multiplication / attention / transition -------------

def _interleave(left, right):
  """[..., h] x2 -> [..., 2h] with a at even and b at odd channels."""
  return np.stack([left, right], -1).reshape(left.shape[:-1] + (-1,))


def _trimul(g, p):
  return {
      'left_norm_input/scale': g(f'{p}.layer_norm_input.weight'),
      'left_norm_input/offset': g(f'{p}.layer_norm_input.bias'),
      'projection/weights': _interleave(g(f'{p}.left_projection.weight'), g(f'{p}.right_projection.weight')),
      'projection/bias': _interleave(g(f'{p}.left_projection.bias'), g(f'{p}.right_projection.bias')),
      'gate/weights': _interleave(g(f'{p}.left_gate.weight'), g(f'{p}.right_gate.weight')),
      'gate/bias': _interleave(g(f'{p}.left_gate.bias'), g(f'{p}.right_gate.bias')),
      'center_norm/scale': g(f'{p}.center_layer_norm.weight'),
      'center_norm/offset': g(f'{p}.center_layer_norm.bias'),
      'output_projection/weights': g(f'{p}.output_projection.weight'),
      'output_projection/bias': g(f'{p}.output_projection.bias'),
      'gating_linear/weights': g(f'{p}.gating_linear.weight'),
      'gating_linear/bias': g(f'{p}.gating_linear.bias'),
  }


def _triatt(g, p):
  q = g(f'{p}.attention.query_w')                          # (c, H, D)
  c, h, d = q.shape
  return {
      'act_norm/scale': g(f'{p}.query_norm.weight'),
      'act_norm/offset': g(f'{p}.query_norm.bias'),
      'pair_bias_projection/weights': g(f'{p}.feat_2d_weights'),
      'q_projection/weights': q.transpose(1, 2, 0),
      'k_projection/weights': g(f'{p}.attention.key_w').transpose(1, 2, 0),
      'v_projection/weights': g(f'{p}.attention.value_w'),
      'gating_query/weights': g(f'{p}.attention.gating_w').reshape(c, h * d).T,
      'gating_query/bias': g(f'{p}.attention.gating_b').reshape(-1),
      'output_projection/weights': g(f'{p}.attention.output_w').reshape(h * d, c),
      'output_projection/bias': g(f'{p}.attention.output_b'),
  }


def _transition(g, p):
  return {
      'input_layer_norm/scale': g(f'{p}.input_layer_norm.weight'),
      'input_layer_norm/offset': g(f'{p}.input_layer_norm.bias'),
      'transition1/weights': np.concatenate([g(f'{p}.proj_a.weight'), g(f'{p}.proj_b.weight')], 1),
      'transition2/weights': g(f'{p}.proj_out.weight'),
  }


def _pair_block(g, p):
  out = {}
  out.update(_nest('triangle_multiplication_outgoing', _trimul(g, f'{p}.triangle_multiplication_outgoing')))
  out.update(_nest('triangle_multiplication_incoming', _trimul(g, f'{p}.triangle_multiplication_incoming')))
  out.update(_nest('pair_attention1', _triatt(g, f'{p}.triangle_attention_starting_node')))
  out.update(_nest('pair_attention2', _triatt(g, f'{p}.triangle_attention_ending_node')))
  out.update(_nest('pair_transition', _transition(g, f'{p}.pair_transition')))
  return out


def _pairformer_block(g, p):
  out = _pair_block(g, p)
  a = f'{p}.single_attention_with_pair_bias'
  c = g(f'{a}.q_lin.weight').shape[0]
  h = g(f'{a}.b_lin.weight').shape[1]
  heads = lambda w: w.reshape(c, h, -1)
  out.update({
      'single_attention_layer_norm/scale': g(f'{a}.ln.weight'),
      'single_attention_layer_norm/offset': g(f'{a}.ln.bias'),
      'single_pair_logits_norm/scale': g(f'{a}.b_ln.weight'),
      'single_pair_logits_norm/offset': g(f'{a}.b_ln.bias'),
      'single_pair_logits_projection/weights': g(f'{a}.b_lin.weight'),
      'single_attention_q_projection/weights': heads(g(f'{a}.q_lin.weight')),
      'single_attention_q_projection/bias': g(f'{a}.q_lin.bias').reshape(h, -1),
      'single_attention_k_projection/weights': heads(g(f'{a}.k_lin.weight')),
      'single_attention_v_projection/weights': heads(g(f'{a}.v_lin.weight')),
      'single_attention_gating_query/weights': g(f'{a}.g_lin.weight'),
      'single_attention_transition2/weights': g(f'{a}.out_lin1.weight'),
  })
  out.update(_nest('single_transition', _transition(g, f'{p}.single_transition')))
  return out


def _msa_block(g, p):
  out = _pair_block(g, p)
  o = f'{p}.outer_product_mean'
  out.update({
      'outer_product_mean/layer_norm_input/scale': g(f'{o}.layer_norm_input.weight'),
      'outer_product_mean/layer_norm_input/offset': g(f'{o}.layer_norm_input.bias'),
      'outer_product_mean/left_projection/weights': g(f'{o}.left_projection.weight'),
      'outer_product_mean/left_projection/bias': g(f'{o}.left_projection.bias'),
      'outer_product_mean/right_projection/weights': g(f'{o}.right_projection.weight'),
      'outer_product_mean/right_projection/bias': g(f'{o}.right_projection.bias'),
      'outer_product_mean/output_w': g(f'{o}.output_w'),
      'outer_product_mean/output_b': g(f'{o}.output_b'),
  })
  m = f'{p}.msa_pair_weighted_averaging'
  v = g(f'{m}.v_proj_w')                                    # (c, H, D)
  c = v.shape[0]
  out.update({
      'msa_attention1/act_norm/scale': g(f'{m}.query_norm.weight'),
      'msa_attention1/act_norm/offset': g(f'{m}.query_norm.bias'),
      'msa_attention1/pair_norm/scale': g(f'{m}.feat_2d_norm.weight'),
      'msa_attention1/pair_norm/offset': g(f'{m}.feat_2d_norm.bias'),
      'msa_attention1/pair_logits/weights': g(f'{m}.bias_proj_w'),
      'msa_attention1/v_projection/weights': v,
      'msa_attention1/gating_query/weights': g(f'{m}.gating_proj_w').reshape(c, -1),
      'msa_attention1/output_projection/weights': g(f'{m}.out_proj_w').reshape(-1, c),
  })
  out.update(_nest('msa_transition', _transition(g, f'{p}.msa_transition')))
  return out


def _count(sd, prefix):
  n = 0
  while any(k.startswith(f'{prefix}.{n}.') for k in sd):
    n += 1
  return n


# --- input embedding, recycling, MSA embedding --------------------------------

def _target_feat(w):
  """449 HF3 input rows [atom 384, restype 32, profile 32, deletion mean 1] ->
  ours [restype 31, profile 31, deletion mean, atom 384, restype DN, profile DN]
  (model.hf3_dn_columns)."""
  r = AF3_TO_HF3
  dn = HF3_UNK_DNA
  return np.concatenate([w[384 + r], w[416 + r], w[448:449], w[0:384],
                         w[384 + dn:385 + dn], w[416 + dn:417 + dn]], 0)


def _trunk_embedding(g, sd, params):
  _put(params, E, {
      'single_activations/weights': _target_feat(g('input_embedder.single_project.weight')),
      'left_single/weights': _target_feat(g('input_embedder.single_to_pair_project.weight')),
      'right_single/weights': _target_feat(g('input_embedder.single_to_pair_project.weight')),
      '~_relative_encoding/position_activations/weights':
          g('input_embedder.relative_position_encoding.rel_position_project.weight'),
      'bond_embedding/weights': g('input_embedder.token_bond_project.weight'),
      'prev_embedding_layer_norm/scale': g(f'{EP}.pair_norm.weight'),
      'prev_embedding_layer_norm/offset': g(f'{EP}.pair_norm.bias'),
      'prev_embedding/weights': g(f'{EP}.pair_project.weight'),
      'prev_single_embedding_layer_norm/scale': g(f'{EP}.single_norm.weight'),
      'prev_single_embedding_layer_norm/offset': g(f'{EP}.single_norm.bias'),
      'prev_single_embedding/weights': g(f'{EP}.single_project.weight'),
      'extra_msa_target_feat/weights': _target_feat(g(f'{EP}.msa_module.single_project.weight')),
  })
  # the MSA one-hot: our 32 rows (31 classes + the mask class, which HF3 lacks
  # and so contributes nothing), then has_deletion and deletion_value
  w = g(f'{EP}.msa_module.msa_project.weight')               # (34, 64)
  msa = np.zeros((34, w.shape[1]), np.float32)
  msa[:31] = w[AF3_TO_HF3]
  msa[32:34] = w[32:34]
  _put(params, E, {'msa_activations/weights': msa,
                   # a DNA column reading our nucleic unknown is HelixFold3's DN, not its RNA N
                   'hf3_dn_msa': w[HF3_UNK_DNA] - w[AF3_TO_HF3[30]]})


# --- atom cross-attention (input embedder, diffusion, confidence) -------------

def _rope_perm(d):
  """Columns of one head, interleaved (2i, 2i+1) -> half-split (i, i + d/2)."""
  return np.concatenate([np.arange(0, d, 2), np.arange(1, d, 2)])


def _atom_transformer(g, p, name, rope):
  """3 blocks of HF3's atom transformer -> the stock AF3 stack under `name`
  (layer-stacked dict) and its shared pair LayerNorm + per-block projection."""
  n = _count(g.sd, f'{p}.diff_transformer.attention_list')
  blocks, logits = [], []
  for i in range(n):
    a = f'{p}.diff_transformer.attention_list.{i}'
    t = f'{p}.diff_transformer.transition_list.{i}'
    blocks.append(_diff_block(g, a, t, name, rope, cross=True))
    logits.append(g(f'{a}.b_ln.weight')[:, None] * g(f'{a}.b_lin.weight'))
  return _stack(blocks), np.stack(logits, 1)                 # (c_p, n, H)


def _adaln(g, p, name):
  return {
      f'{name}single_cond_layer_norm/scale': g(f'{p}.s_ln.weight'),
      f'{name}single_cond_scale/weights': g(f'{p}.lin1.weight'),
      f'{name}single_cond_scale/bias': g(f'{p}.lin1.bias'),
      f'{name}single_cond_bias/weights': g(f'{p}.lin2.weight'),
  }


def _diff_block(g, a, t, name, rope, cross):
  """One HF3 attention + conditioned transition block. `cross` writes the adaLN
  to both the query and the key copies (HF3 runs one adaLN, then gathers keys
  from it: per atom, so the same function)."""
  d = {}
  if cross:
    d.update(_adaln(g, f'{a}.ln', f'{name}q'))
    d.update(_adaln(g, f'{a}.ln', f'{name}k'))
  else:
    d.update(_adaln(g, f'{a}.ln', name))
  h = g(f'{a}.b_lin.weight').shape[1]
  qw = g(f'{a}.q_lin.weight')
  c = qw.shape[0]
  dh = qw.shape[1] // h
  perm = _rope_perm(dh) if rope else np.arange(dh)
  heads = lambda w: w.reshape(c, h, dh)[..., perm]
  d.update({
      f'{name}q_projection/weights': heads(qw),
      f'{name}q_projection/bias': g(f'{a}.q_lin.bias').reshape(h, dh)[..., perm],
      f'{name}k_projection/weights': heads(g(f'{a}.k_lin.weight')),
      f'{name}v_projection/weights': g(f'{a}.v_lin.weight').reshape(c, h, dh),
      f'{name}gating_query/weights': g(f'{a}.g_lin.weight'),
      f'{name}transition2/weights': g(f'{a}.out_lin1.weight'),
      f'{name}adaptive_zero_cond/weights': g(f'{a}.out_lin2.weight'),
      f'{name}adaptive_zero_cond/bias': g(f'{a}.out_lin2.bias'),
  })
  d.update(_adaln(g, f'{t}.ln', f'{name}ffw_'))
  d.update({
      f'{name}ffw_transition1/weights': np.concatenate([g(f'{t}.lin1.weight'), g(f'{t}.lin2.weight')], 1),
      f'{name}ffw_transition2/weights': g(f'{t}.lin4.weight'),
      f'{name}ffw_adaptive_zero_cond/weights': g(f'{t}.lin3.weight'),
      f'{name}ffw_adaptive_zero_cond/bias': g(f'{t}.lin3.bias'),
  })
  return d


def _atom_meta(g, p, name):
  w = g(f'{p}.lin_atom_meta_to_cond_feat.weight')             # (389, 128)
  return {
      f'{name}_embed_ref_pos/weights': w[0:3],
      f'{name}_embed_ref_charge/weights': w[3:4],
      f'{name}_embed_ref_mask/weights': w[4:5],
      f'{name}_embed_ref_element/weights': w[5:133],
      f'{name}_embed_ref_atom_name/weights': w[133:389],
  }


def _atom_pair(g, p, name):
  # one Linear for rows and columns in HF3, copied to both of ours
  out = {
      f'{name}_embed_pair_offsets/weights': g(f'{p}.lin_pos_offset_to_apair.weight'),
      f'{name}_embed_pair_distances/weights': g(f'{p}.lin_inv_sq_dist_to_apair.weight'),
      f'{name}_single_to_pair_cond_row/weights': g(f'{p}.lin_single_cond_to_pair_repr.weight'),
      f'{name}_single_to_pair_cond_col/weights': g(f'{p}.lin_single_cond_to_pair_repr.weight'),
  }
  out[f'{name}_embed_pair_offsets_valid/weights'] = g(f'{p}.lin_valid_mask_to_apair.weight')
  for i, j in ((1, 1), (2, 3), (3, 5)):
    out[f'{name}_pair_mlp_{i}/weights'] = g(f'{p}.mlp_pair_active.{j}.weight')
  out[f'{name}_project_atom_features_for_aggr/weights'] = g(f'{p}.lin_atom_to_token.weight')
  return out


def _atom_encoder(g, params, p, scope, name, rope, conditioned):
  """An HF3 AtomAttentionEncoder -> `{scope}/{name}_*` params."""
  local = {**_atom_meta(g, p, name), **_atom_pair(g, p, name)}
  if conditioned:
    local.update({
        f'{name}_lnorm_trunk_single_cond/scale': g(f'{p}.ln_trunk_single_to_cond_atom_feat.weight'),
        f'{name}_lnorm_trunk_single_cond/offset': g(f'{p}.ln_trunk_single_to_cond_atom_feat.bias'),
        f'{name}_embed_trunk_single_cond/weights': g(f'{p}.lin_trunk_single_to_cond_atom_feat.weight'),
        f'{name}_lnorm_trunk_pair_cond/scale': g(f'{p}.ln_cond_pair_feat_to_pair_repr.weight'),
        f'{name}_lnorm_trunk_pair_cond/offset': g(f'{p}.ln_cond_pair_feat_to_pair_repr.bias'),
        f'{name}_embed_trunk_pair_cond/weights': g(f'{p}.lin_cond_pair_feat_to_pair_repr.weight'),
        f'{name}_atom_positions_to_features/weights': g(f'{p}.lin_noise_pos_to_single_repr.weight'),
    })
  _put(params, scope, local)
  enc = f'{name}_atom_transformer_encoder'
  stacked, logits = _atom_transformer(g, f'{p}.atom_transformer', enc, rope)
  _put(params, f'{scope}/{enc}/__layer_stack_with_per_layer', stacked)
  _put(params, f'{scope}/{enc}', {
      'pair_input_layer_norm/scale': np.ones(logits.shape[0], np.float32),
      'pair_logits_projection/weights': logits,
  })


def _atom_decoder(g, params, p, scope, name, rope):
  dec = f'{name}_atom_transformer_decoder'
  stacked, logits = _atom_transformer(g, f'{p}.atom_transformer', dec, rope)
  _put(params, f'{scope}/{dec}/__layer_stack_with_per_layer', stacked)
  _put(params, f'{scope}/{dec}', {
      'pair_input_layer_norm/scale': np.ones(logits.shape[0], np.float32),
      'pair_logits_projection/weights': logits,
  })
  _put(params, scope, {
      f'{name}_project_token_features_for_broadcast/weights': g(f'{p}.lin0.weight'),
      f'{name}_atom_features_layer_norm/scale': g(f'{p}.ln1.weight'),
      f'{name}_atom_features_layer_norm/offset': g(f'{p}.ln1.bias'),
      f'{name}_atom_features_to_position_update/weights': g(f'{p}.lin1.weight'),
  })


# --- diffusion module ----------------------------------------------------------

def _diffusion(g, params):
  dm = 'diffusion_module'
  dc = f'{dm}.diffusion_conditioning'
  single_ln = _reorder_features_1d(np.stack([g(f'{dc}.single_ln1.weight'), g(f'{dc}.single_ln1.bias')], 1),
                                   pad_unk_dna=True, remap=AF3_TO_HF3, unk=HF3_UNK_DNA)
  local = {
      'relpe_projection/weights': g('diff_rel_pos_encoding.rel_position_project.weight'),
      'pair_cond_initial_norm/scale': g(f'{dc}.pair_ln.weight'),
      'pair_cond_initial_norm/offset': g(f'{dc}.pair_ln.bias'),
      'pair_cond_initial_projection/weights': g(f'{dc}.pair_lin.weight'),
      'single_cond_initial_norm/scale': single_ln[:, 0],
      'single_cond_initial_norm/offset': single_ln[:, 1],
      'single_cond_initial_projection/weights': _reorder_features_1d(
          g(f'{dc}.single_lin1.weight'), pad_unk_dna=True, remap=AF3_TO_HF3, unk=HF3_UNK_DNA),
      'noise_embedding_initial_norm/scale': g(f'{dc}.single_ln2.weight'),
      'noise_embedding_initial_norm/offset': g(f'{dc}.single_ln2.bias'),
      'noise_embedding_initial_projection/weights': g(f'{dc}.single_lin2.weight'),
      'fourier_embedding_weight': g(f'{dc}.fourier_embedding.w'),
      'fourier_embedding_bias': g(f'{dc}.fourier_embedding.b'),
      'single_cond_embedding_norm/scale': g(f'{dm}.ln1.weight'),
      'single_cond_embedding_norm/offset': g(f'{dm}.ln1.bias'),
      'single_cond_embedding_projection/weights': g(f'{dm}.lin1.weight'),
      'output_norm/scale': g(f'{dm}.ln2.weight'),
      'output_norm/offset': g(f'{dm}.ln2.bias'),
  }
  for kind in ('pair', 'single'):
    for idx, n in ((0, 1), (1, 2)):
      tr = f'{dc}.{kind}_trans{n}'
      local.update({
          f'{kind}_transition_{idx}ffw_layer_norm/scale': g(f'{tr}.ln.weight'),
          f'{kind}_transition_{idx}ffw_layer_norm/offset': g(f'{tr}.ln.bias'),
          f'{kind}_transition_{idx}ffw_transition1/weights':
              np.concatenate([g(f'{tr}.lin1.weight'), g(f'{tr}.lin2.weight')], 1),
          f'{kind}_transition_{idx}ffw_transition2/weights': g(f'{tr}.lin3.weight'),
      })
  _put(params, DH, local)

  # token transformer: 24 blocks as (6 super-blocks, 4), the stock AF3 path
  dt = f'{dm}.diffusion_transformer'
  n = _count(g.sd, f'{dt}.attention_list')
  blocks, logits = [], []
  for i in range(n):
    a, t = f'{dt}.attention_list.{i}', f'{dt}.transition_list.{i}'
    blocks.append(_diff_block(g, a, t, 'transformer', rope=True, cross=False))
    logits.append(g(f'{a}.b_ln.weight')[:, None] * g(f'{a}.b_lin.weight'))
  sb = 4
  stacked = {k: v.reshape((n // sb, sb) + v.shape[1:]) for k, v in _stack(blocks).items()}
  _put(params, f'{DH}/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer', stacked)
  lg = np.stack(logits).reshape(n // sb, sb, *logits[0].shape)  # (6, 4, c, H)
  _put(params, f'{DH}/transformer/__layer_stack_with_per_layer', {
      'pair_logits_projection/weights': lg.transpose(0, 2, 1, 3)})  # (6, c, 4, H)
  _put(params, f'{DH}/transformer', {
      'pair_input_layer_norm/scale': np.ones(lg.shape[2], np.float32)})

  _atom_encoder(g, params, f'{dm}.atom_encoder', DH, 'diffusion', rope=True, conditioned=True)
  _atom_decoder(g, params, f'{dm}.atom_decoder', DH, 'diffusion', rope=True)


# --- whole model ---------------------------------------------------------------

def map_helixfold3_to_af3(sd):
  g = _Reader(sd)
  params = {}
  _trunk_embedding(g, sd, params)
  _atom_encoder(g, params, 'input_embedder.atom_attention_encoder', 'diffuser',
                'evoformer_conditioning', rope=False, conditioned=False)
  n_pf = _count(sd, f'{EP}.pairformer_stack')
  _put(params, PF, _stack([_pairformer_block(g, f'{EP}.pairformer_stack.{i}') for i in range(n_pf)]))
  n_msa = _count(sd, f'{EP}.msa_module.evoformer_stack')
  _put(params, MS, _stack([_msa_block(g, f'{EP}.msa_module.evoformer_stack.{i}') for i in range(n_msa)]))
  te = f'{EP}.template_embedder'
  _put(params, TE, {
      'z_norm/scale': g(f'{te}.pair_norm.weight'),
      'z_norm/offset': g(f'{te}.pair_norm.bias'),
      'z_proj/weights': g(f'{te}.pair_project.weight'),
      'a_proj/weights': g(f'{te}.template_project.weight'),   # HelixFold3TemplateEmbedding's own order
      'v_norm/scale': g(f'{te}.out_norm.weight'),
      'v_norm/offset': g(f'{te}.out_norm.bias'),
      'u_proj/weights': g(f'{te}.out_projection.weight'),
  })
  n_t = _count(sd, f'{te}.pairformer_stack')
  _put(params, TS, _stack([_pair_block(g, f'{te}.pairformer_stack.{i}') for i in range(n_t)]))
  _put(params, 'diffuser/distogram_head', {
      'half_logits/weights': g('distogram.half_logits.weight'),
      'half_logits/bias': g('distogram.half_logits.bias'),
  })
  _diffusion(g, params)
  _confidence(g, sd, params)
  return params


def _confidence(g, sd, params):
  ch = 'confidence_head'
  n = _count(sd, f'{ch}.pairformer_stack')
  _put(params, CP, _stack([_pairformer_block(g, f'{ch}.pairformer_stack.{i}') for i in range(n)]))
  _put(params, CH, {
      'pae_logits_ln/scale': g(f'{ch}.ln_pae.weight'),
      'pae_logits_ln/offset': g(f'{ch}.ln_pae.bias'),
      'pae_logits/weights': g(f'{ch}.lin_pae.weight'),
      'pae_logits/bias': g(f'{ch}.lin_pae.bias'),
      'logits_ln/scale': g(f'{ch}.ln_pde.weight'),
      'logits_ln/offset': g(f'{ch}.ln_pde.bias'),
      'left_half_distance_logits/weights': g(f'{ch}.lin_pde.weight'),
      'left_half_distance_logits/bias': g(f'{ch}.lin_pde.bias'),
  })
  # HelixFold3's re-embedding of the predicted structure: the single->pair
  # terms over LayerNorm(833) [conf atom encoder 384 | s_inputs 449], the
  # centre-atom distogram (45 classes), and the per-atom pLDDT / resolved
  # heads behind its own atom decoder (network/confidence_head.py
  # `_helixfold3_*`). lin_s_left adds along rows -> our right_*, and vice versa.
  ln_s = np.stack([g(f'{ch}.ln_s.weight'), g(f'{ch}.ln_s.bias')], 1)
  _put(params, CH, {
      'hf3_single_norm/scale': _conf_single(ln_s[:, 0]),
      'hf3_single_norm/offset': _conf_single(ln_s[:, 1]),
      'right_target_feat_project/weights': _conf_single(g(f'{ch}.lin_s_left.weight')),
      'right_target_feat_project/bias': g(f'{ch}.lin_s_left.bias'),
      'left_target_feat_project/weights': _conf_single(g(f'{ch}.lin_s_right.weight')),
      'left_target_feat_project/bias': g(f'{ch}.lin_s_right.bias'),
      'distogram_feat_project/weights': g(f'{ch}.lin_dij.weight'),
      'distogram_feat_project/bias': g(f'{ch}.lin_dij.bias'),
  })
  _atom_encoder(g, params, f'{ch}.atom_encoder', CH, 'confidence', rope=False, conditioned=True)
  _atom_decoder_conf(g, params, f'{ch}.atom_decoder')
  # lin1 (no bias) then lin_plddt / lin_resolved: two Linears in a row
  w1 = g(f'{ch}.atom_decoder.lin1.weight')
  _put(params, CH, {
      'hf3_single_ln/scale': g(f'{ch}.ln_si.weight'),
      'hf3_single_ln/offset': g(f'{ch}.ln_si.bias'),
      'hf3_plddt_logits/weights': w1 @ g(f'{ch}.lin_plddt.weight'),
      'hf3_plddt_logits/bias': g(f'{ch}.lin_plddt.bias'),
      'hf3_resolved_logits/weights': w1 @ g(f'{ch}.lin_resolved.weight'),
      'hf3_resolved_logits/bias': g(f'{ch}.lin_resolved.bias'),
  })


def _conf_single(w):
  """LayerNorm(833)'s rows [conf atom encoder 384 | s_inputs 449] -> ours
  [conf atom encoder 384 | target_feat 447 | the two unknown-DNA columns]."""
  return np.concatenate([w[:384], _target_feat(w[384:])], 0)


def _atom_decoder_conf(g, params, p):
  dec = 'confidence_atom_transformer_decoder'
  stacked, logits = _atom_transformer(g, f'{p}.atom_transformer', dec, rope=False)
  _put(params, f'{CH}/{dec}/__layer_stack_with_per_layer', stacked)
  _put(params, f'{CH}/{dec}', {
      'pair_input_layer_norm/scale': np.ones(logits.shape[0], np.float32),
      'pair_logits_projection/weights': logits,
  })
  _put(params, CH, {
      'confidence_project_token_features_for_broadcast/weights': g(f'{p}.lin0.weight'),
      'confidence_atom_features_layer_norm/scale': g(f'{p}.ln1.weight'),
      'confidence_atom_features_layer_norm/offset': g(f'{p}.ln1.bias'),
  })


# Tensors no forward reads: the input embedder's atom encoder is called with no
# trunk and no positions (modules_all_atom.py:264), so its trunk-conditioning
# and position Linears are untrained leftovers of the shared class.
DEAD = tuple(f'input_embedder.atom_attention_encoder.{k}' for k in (
    'ln_trunk_single_to_cond_atom_feat.weight', 'ln_trunk_single_to_cond_atom_feat.bias',
    'lin_trunk_single_to_cond_atom_feat.weight', 'ln_cond_pair_feat_to_pair_repr.weight',
    'ln_cond_pair_feat_to_pair_repr.bias', 'lin_cond_pair_feat_to_pair_repr.weight',
    'lin_noise_pos_to_single_repr.weight'))


def _check_dims(sd, model_name='helixfold3'):
  from alphafold3.model import model
  from alphafold3.model import model_registry
  cfg = model_registry.get(model_name).configure(model.Model.Config())
  got = {
      'pairformer': _count(sd, f'{EP}.pairformer_stack'),
      'msa': _count(sd, f'{EP}.msa_module.evoformer_stack'),
      'template': _count(sd, f'{EP}.template_embedder.pairformer_stack'),
      'diffusion_token': _count(sd, 'diffusion_module.diffusion_transformer.attention_list'),
      'confidence': _count(sd, 'confidence_head.pairformer_stack'),
  }
  want = {
      'pairformer': cfg.evoformer.pairformer.num_layer,
      'msa': cfg.evoformer.msa_stack.num_layer,
      'template': cfg.evoformer.template.template_stack.num_layer,
      'diffusion_token': cfg.heads.diffusion.transformer.num_blocks,
      'confidence': cfg.heads.confidence.pairformer.num_layer,
  }
  bad = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
  if bad:
    raise ValueError('helixfold3 checkpoint does not match the graph: ' + ', '.join(
        f'{k} has {a} blocks, the graph {b}' for k, (a, b) in bad.items()))


def convert_helixfold3_weights(checkpoint, output_dir):
  """Convert HelixFold3-20250714.pdparams to a loadable AF3-haiku blob."""
  sd = load_checkpoint(checkpoint)
  _check_dims(sd)
  return Path(C.write_params_blob(output_dir, 'helixfold3.bin.zst',
                                  map_helixfold3_to_af3(sd), add_meta=True))
