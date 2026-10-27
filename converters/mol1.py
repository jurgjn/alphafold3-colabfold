"""Mol-1 (Molence Bio, github.com/molence/Mol-1) -> AF3-haiku blob.

Mol-1 is a fork of Protenix at upstream bd54a05 -- v2.0.0's model code -- so
everything outside its trunk converts with converters/protenix2's mappers. The
trunk is what differs (mol/model/modules/pairformer.py PairformerStack):

  pairformer_stack.blocks.0..7    PairMixer: a pairformer block with no triangle
                                  attention, at the full pair width (448)
  adapters.before_block_8         bias-free Linear 448 -> 320
  pairformer_stack.blocks.8..47   ordinary pairformer blocks at 320
  adapters.after_blocks           bias-free Linear 320 -> 448

which the graph builds as Evoformer.pairmixer (its own layer stack, called
first, so `__layer_stack_no_per_layer_1`), trunk_pair_in / trunk_pair_out, and
the trunk at `__layer_stack_no_per_layer_2`. Triangle-attention heads are
c_z // 32 per stack -- 14 at 448 (MSA pair stack, confidence pairformer), 10
in the 320 trunk -- and are read off the checkpoint, never assumed.
"""

import re

from . import common as C
from . import protenix2 as P

_MIXER_SCOPE = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairmixer'
_TRUNK_SCOPE = 'diffuser/evoformer/__layer_stack_no_per_layer_2/trunk_pairformer'


def _count(sd, prefix):
  idx = {int(m.group(1)) for k in sd if (m := re.match(rf'{re.escape(prefix)}\.(\d+)\.', k))}
  return (max(idx) + 1) if idx else 0


def derive_dims(sd):
  """Mol-1's shape, read off the checkpoint: the split trunk and per-stack heads.

  Not protenix2.derive_dims: that reads the trunk head count off block 0's
  triangle attention, which a PairMixer block does not have."""
  n_blocks = _count(sd, 'pairformer_stack.blocks')
  # a PairMixer block is exactly a block without triangle attention
  mixer = [i for i in range(n_blocks)
           if not P._has(sd, f'pairformer_stack.blocks.{i}.tri_att_start.linear.weight')]
  n_mixer = len(mixer)
  if mixer != list(range(n_mixer)):
    raise ValueError(f'Mol-1: the PairMixer blocks are not the leading ones: {mixer}')
  head = lambda key: int(P._get(sd, key).shape[0])
  return dict(
      n_mixer=n_mixer,
      n_trunk=n_blocks - n_mixer,
      c_z=head('linear_no_bias_z_cycle.weight'),
      c_trunk=head(f'pairformer_stack.blocks.{n_mixer}.tri_mul_out.linear_a_p.weight'),
      trunk_H=head(f'pairformer_stack.blocks.{n_mixer}.tri_att_start.linear.weight'),
      msa_H=head('msa_module.blocks.0.pair_stack.tri_att_start.linear.weight'),
      conf_H=head('confidence_head.pairformer_stack.blocks.0.tri_att_start.linear.weight'),
      templ_H=head('template_embedder.pairformer_stack.blocks.0.tri_att_start.linear.weight'),
      n_msa=_count(sd, 'msa_module.blocks'),
      n_template=_count(sd, 'template_embedder.pairformer_stack.blocks'),
      n_confidence=_count(sd, 'confidence_head.pairformer_stack.blocks'),
      n_diff_token=_count(sd, 'diffusion_module.diffusion_transformer.blocks'),
      n_diff_atom_enc=_count(sd, 'diffusion_module.atom_attention_encoder'
                                 '.atom_transformer.diffusion_transformer.blocks'),
      n_input_atom_enc=_count(sd, 'input_embedder.atom_attention_encoder'
                                  '.atom_transformer.diffusion_transformer.blocks'),
  )


def _pairmixer_block(sd, i, single_H, single_D):
  """A PairMixer block: Protenix's block leaves, less tri_att_start / tri_att_end."""
  prefix = f'pairformer_stack.blocks.{i}'
  out = {}
  for tag, name, outgoing in [('triangle_multiplication_outgoing', 'tri_mul_out', True),
                              ('triangle_multiplication_incoming', 'tri_mul_in', False)]:
    for k, v in C.triangle_mul(sd, f'{prefix}.{name}', P.D, outgoing=outgoing).items():
      out[f'{tag}/{k}'] = v
  for k, v in C.transition(sd, f'{prefix}.pair_transition', P.D).items():
    out[f'pair_transition/{k}'] = v
  out.update(C.single_attention(sd, f'{prefix}.attention_pair_bias', P.D, single_H, single_D))
  for k, v in C.transition(sd, f'{prefix}.single_transition', P.D).items():
    out[f'single_transition/{k}'] = v
  return out


def map_mol1_to_af3(sd):
  """Convert Mol-1's state dict (module.* stripped) to AF3 haiku params."""
  d = pd = derive_dims(sd)
  params = {}
  P.map_input_embeddings(sd, params)
  P.map_msa_stack(sd, params, n_blocks=pd['n_msa'], pair_H=d['msa_H'])
  # the PairMixer stage, then the trunk with blocks n_mixer.. renumbered from 0
  P._populate_scope(params, _MIXER_SCOPE, P._stack_blocks(
      lambda i: _pairmixer_block(sd, i, single_H=16, single_D=24), d['n_mixer']))
  P._populate_scope(params, _TRUNK_SCOPE, P._stack_blocks(
      lambda i: P._pairformer_block(sd, i + d['n_mixer'], d['trunk_H'], 32, 16, 24),
      d['n_trunk']))
  for name, key in (('trunk_pair_in', f'before_block_{d["n_mixer"]}'), ('trunk_pair_out', 'after_blocks')):
    P._set(params, f'diffuser/evoformer/{name}', 'weights',
           P._t(P._get(sd, f'pairformer_stack.adapters.{key}.pair_adapter.weight')))
  P.map_confidence_head(sd, params, n_layers=pd['n_confidence'], pair_H=d['conf_H'])
  P.map_distogram_head(sd, params)
  P.map_template_embedder(sd, params, n_blocks=pd['n_template'], templ_H=pd['templ_H'])
  P.map_evoformer_conditioning(sd, params, n_atom=pd['n_input_atom_enc'])
  P.map_diffusion(sd, params, n_token=pd['n_diff_token'],
                  n_super=max(1, pd['n_diff_token'] // 4), n_atom=pd['n_diff_atom_enc'])
  return params


DEAD_TENSORS = P.DEAD_TENSORS
# Mol-1 checkpoints are Protenix checkpoints: {'model': state dict with `module.` keys, ...}
load_protenix_checkpoint = P.load_protenix_checkpoint


def convert_mol1_weights(checkpoint, output_dir):
  """Convert mol-1-base.pt to a loadable AF3-haiku blob dir.

  Refuses a checkpoint whose split trunk or widths differ from the registry's
  MOL1_SETTINGS: the graph is built from those, the blob from the checkpoint, and
  a mismatch between them loads silently and is wrong everywhere.
  """
  from alphafold3.model import model as af3_model
  from alphafold3.model import model_registry
  sd = P.load_protenix_checkpoint(checkpoint)
  d = derive_dims(sd)
  cfg = model_registry.get('mol1').configure(af3_model.Model.Config())
  e = cfg.evoformer
  want = dict(n_mixer=e.pairmixer.num_layer, n_trunk=e.pairformer.num_layer, c_z=e.pair_channel,
              c_trunk=e.trunk_pair_channel, trunk_H=e.pairformer.pair_attention.num_head,
              msa_H=e.msa_stack.pair_attention.num_head,
              conf_H=cfg.heads.confidence.pairformer.pair_attention.num_head)
  bad = {k: (d[k], v) for k, v in want.items() if d[k] != v}
  if bad:
    raise ValueError('mol1 checkpoint does not match the graph: ' + ', '.join(
        f'{k} is {got} but the graph wants {exp}' for k, (got, exp) in bad.items()))
  return C.write_params_blob(output_dir, 'mol1.bin.zst', map_mol1_to_af3(sd), add_meta=True)
