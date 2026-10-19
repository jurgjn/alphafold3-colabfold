"""The STOCK RoseTTAFold3 half of vendor_native_parity.py (rosettafold3).

  python rosettafold3_upstream_runner.py featurise rosettafold3 CASE.json OUT_DIR
  python rosettafold3_upstream_runner.py forward   rosettafold3 OUT_DIR IN.npz OUT.npz

Run in ~/venv_rf3 with ONLY foundry's two source trees on PYTHONPATH
(~/foundry_rf3/src, ~/foundry_rf3/models/rf3/src; see VENDORS for the env).
Everything is RF3's own, driven the way `rf3 fold inputs=X.json` drives it
(rf3/cli.py -> rf3/inference.py): an `RF3InferenceEngine` built with the CLI's
engine arguments (configs/inference_engine/rf3.yaml), which loads the
checkpoint's OWN train_cfg (model, EMA wrapper, atomworks transform pipeline of
the first validation dataset) and overrides only what the engine overrides;
then `engine.run(inputs=<the JSON path>)` -- prepare_inference_inputs_from_paths
-> InferenceInput.from_json_dict -> engine.pipeline -> trainer.validation_step
-> model.forward (the EMA shadow) -> compile_af3_style_confidence_outputs.

The single override is the stochastic sampler,
`SampleDiffusion.sample_diffusion_like_af3`: it returns the injected coordinates
(which the confidence head then scores) and runs the real diffusion module ONCE
on an injected (noised coordinates, sigma). See boltz2_upstream_runner.py for
why nothing else is hand-built.

RANDOM DRAWS. The engine seeds everything at construction (seed=1, as
`rf3 fold seed=1`), then builds the model, then featurises, then predicts --
one process, that order, in both subcommands, so `featurise` sees the draw
`forward` will. The reference conformer (atomworks' RDKit conformer + random
roto-translation) is such a draw; layout.npz carries it (feat.ref_pos,
feat.ref_space_uid) for the harness to inject into ours, and `forward` CHECKS
it re-drew the identical one. The per-recycle MSA subsample
(FeaturizeMSALikeAF3, n_msa=1024) is a draw too, but no case here has more than
1024 rows, so it is a row permutation, which the MSA module is invariant to up
to summation order.
"""
import json
import os
import sys

# cuEquivariance's fused triangle kernels are chosen at `import foundry` if the
# package imports (foundry/__init__.py). An implementation choice, not the
# model, and not installed in ~/venv_rf3 (the `rf3` extra was skipped); the
# variable makes the reference torch path explicit even if it ever is.
os.environ.setdefault('DISABLE_CUEQUIVARIANCE', '1')

import numpy as np

CKPT = os.path.expanduser(os.environ.get(
    'RF3_CKPT', '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt'))
SEED = 1

_THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
          'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
          'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
          'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
          'X': 'UNK'}  # an unknown amino acid is the CCD UNK, as RF3 reads X


def _trivial_msa(msa, seq):
  return (not msa) or msa.strip() == '>q\n%s' % seq


def _rf3_json(j, msa_dir):
  """AF3 fold-input JSON -> RF3's JSON input (a one-element list).

  RF3's schema is atomworks' component list (models/rf3/tests/data/*.json):
  {'seq', 'chain_type', 'chain_id', 'msa_path'} for a polymer, a modified
  residue written inline as `(CCD)`, {'ccd_code'} / {'smiles'} for a ligand,
  and bonds as 'CHAIN/RES_NAME/RES_ID/ATOM' pairs.
    * An AF3 ligand is a CCD (or SMILES) ligand -- MG included: AF3 has no ion
      type, and RF3's JSON has none either (glke_with_ligands.json: ccd_code MG).
    * A chain with no alignment (none, or the query alone) gets NO msa_path,
      which is RF3's single-sequence path as `rf3 fold` runs it with no
      LOCAL_MSA_DIRS: LoadPolymerMSAs finds nothing and the MSA is the query.
    * An unpaired alignment goes to an .a3m file named by msa_path, the way the
      CLI reads one. (AF3's pairedMsa is '' in every case here.)
  """
  comps, resname = [], {}
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    if kind in ('protein', 'rna', 'dna'):
      seq = list(e['sequence'])
      mods = e.get('modifications', [])
      for m in mods:
        pos = m.get('ptmPosition', m.get('basePosition'))
        seq[pos - 1] = '(%s)' % m.get('ptmType', m.get('modificationType'))
      names = []
      for k, c in enumerate(e['sequence']):
        if seq[k].startswith('('):
          names.append(seq[k][1:-1])
        elif kind == 'protein':
          names.append(_THREE[c])
        else:
          names.append(c if kind == 'rna' else 'D' + c)
      for cid in ids:
        resname[cid] = names
      d = {'seq': ''.join(seq), 'chain_id': ids if len(ids) > 1 else ids[0]}
      if kind == 'rna':
        d['chain_type'] = 'POLYRIBONUCLEOTIDE'
      elif kind == 'dna':
        d['chain_type'] = 'POLYDEOXYRIBONUCLEOTIDE'
      else:
        d['chain_type'] = 'POLYPEPTIDE(L)'
      if e.get('templates'):
        raise NotImplementedError('templates: rf3 reads them from a CIF input')
      if not _trivial_msa(e.get('unpairedMsa'), e['sequence']):
        p = os.path.join(msa_dir, '%s.a3m' % ids[0])
        with open(p, 'w') as f:
          f.write(e['unpairedMsa'])
        d['msa_path'] = p
      if not _trivial_msa(e.get('pairedMsa'), e['sequence']):
        raise NotImplementedError('pairedMsa')
      comps.append(d)
    elif kind == 'ligand':
      d = {'chain_id': ids if len(ids) > 1 else ids[0]}
      if 'smiles' in e:
        d['smiles'] = e['smiles']
        names = ['L:0']
      else:
        assert len(e['ccdCodes']) == 1, 'multi-component ligand'
        d['ccd_code'] = e['ccdCodes'][0]
        names = [e['ccdCodes'][0]]
      for cid in ids:
        resname[cid] = names
      comps.append(d)
    else:
      raise NotImplementedError(kind)
  out = {'name': j['name'], 'components': comps}
  if j.get('bondedAtomPairs'):
    out['bonds'] = [['%s/%s/%d/%s' % (a[0], resname[a[0]][a[1] - 1], a[1], a[2]),
                     '%s/%s/%d/%s' % (b[0], resname[b[0]][b[1] - 1], b[1], b[2])]
                    for a, b in j['bondedAtomPairs']]
  return [out]


def _precision():
  import torch
  # TF32: foundry's train.py/validate.py set float32_matmul_precision('medium')
  # (TF32 matmuls); the inference entry point does not, but a library default
  # must not decide it either. A precision setting, not the model: fp32
  # matmuls everywhere for the strict comparison (ours runs at 'highest').
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')


def _engine(n_recycles, precision):
  """`rf3 fold`'s engine (configs/inference_engine/rf3.yaml), constructed as
  rf3/inference.py instantiates it, with these departures, each argued:
    * n_recycles: the harness's recycling_steps + 1 (RF3 counts trunk passes;
      validation_step reads n_cycle from msa_stack.shape[0]).
    * diffusion_batch_size=1: one injected sample is scored.
    * early_stopping_plddt_threshold=None: the CLI's 0.5 only WITHHOLDS the
      output when the first-pass pLDDT is low; it changes no value.
    * metrics_cfg=None: ptm/iptm/clash summaries, not model outputs.
    * trainer precision: the checkpoint trains/infers 'bf16-mixed' (Fabric
      autocast). fp32 for the strict comparison ('32-true'), bf16-mixed when
      the harness asks for bf16 -- a precision choice, not the model.
  """
  from rf3.inference_engines.rf3 import RF3InferenceEngine
  return RF3InferenceEngine(
      ckpt_path=CKPT, n_recycles=n_recycles, diffusion_batch_size=1,
      num_steps=50, template_noise_scale=1e-5,
      raise_if_missing_msa_for_protein_of_length_n=None,
      fallback_conformer_to_input_coords=True, compress_outputs=False,
      early_stopping_plddt_threshold=None, metrics_cfg=None, seed=SEED,
      verbose=False, num_nodes=1, devices_per_node=1,
      trainer_overrides={'precision': '32-true' if precision == 'fp32'
                         else 'bf16-mixed'})


def _unforce_bf16(engine, precision):
  # RF3's two attention-with-pair-bias classes (pairformer_layers.py:241,
  # af3_diffusion_transformer.py:402) hard-code `force_bfloat16 = True` in
  # __init__ -- no config reaches it -- and cast their input to bf16 before
  # q/k/v. A precision choice, not the model (and in a true fp32 run it cannot
  # even execute: a bf16 input meets fp32 weights). Off for fp32, as stock for
  # bf16.
  n = 0
  for m in engine.trainer.state['model'].modules():
    if hasattr(m, 'force_bfloat16'):
      m.force_bfloat16 = precision != 'fp32'
      n += 1
  assert n, 'no force_bfloat16 attribute found -- has RF3 moved it?'


class _Stop(Exception):
  pass


def _run(out_dir, n_recycles, precision, on_pipeline, patch=None):
  """Construct, seed, build, then engine.run -- the CLI's order. on_pipeline
  sees the featurised example; patch (forward only) installs the overrides."""
  _precision()
  engine = _engine(n_recycles, precision)
  engine.initialize()
  _unforce_bf16(engine, precision)
  if patch is not None:
    patch(engine)
  pipe = engine.pipeline

  def pipeline(x):
    out = pipe(x)
    on_pipeline(out)
    return out
  engine.pipeline = pipeline
  try:
    engine.run(inputs=os.path.join(out_dir, 'rf3.json'), out_dir=None)
  except _Stop:
    pass
  return engine


def _layout(ex):
  f = ex['feats']
  aa = ex['atom_array']
  ref_pos = f['ref_pos'].detach().cpu().numpy().copy()
  assert len(aa) == ref_pos.shape[0], (len(aa), ref_pos.shape)
  n_tok = f['restype'].shape[0]
  return dict(
      atom_name=np.array([str(x) for x in aa.atom_name]),
      atom_token=f['atom_to_token_map'].detach().cpu().numpy().copy(),
      atom_pad=np.ones(len(aa), bool),
      token_pad=np.ones(n_tok, bool),
      **{'feat.ref_pos': ref_pos,
         'feat.ref_space_uid':
             f['ref_space_uid'].detach().cpu().numpy().copy(),
         'feat.ref_element':
             f['ref_element'].detach().cpu().numpy().copy()})


def featurise(model, case_json, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  msa_dir = os.path.join(out_dir, 'msa')
  os.makedirs(msa_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'rf3.json'), 'w') as f:
    json.dump(_rf3_json(j, msa_dir), f, indent=1)

  def on_pipeline(ex):
    np.savez(os.path.join(out_dir, 'layout.npz'), **_layout(ex))
    raise _Stop
  # n_recycles=1: the recycle count only sets how many MSA subsamples the
  # pipeline draws, AFTER the reference conformer (atomworks order:
  # GetAF3ReferenceMoleculeFeatures precedes FeaturizeMSALikeAF3), and forward
  # checks the ref_pos it draws against this one.
  _run(out_dir, 1, 'fp32', on_pipeline)


def forward(model, out_dir, in_npz, out_npz):
  import torch
  from rf3.diffusion_samplers import inference_sampler as smod
  import rf3.inference_engines.rf3 as emod
  z = np.load(in_npz)
  precision = 'bf16' if str(z['precision']) == 'bf16' else 'fp32'
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  got, taps, ex_box = {}, {}, {}
  sigma = float(z['noise_level'])

  # THE override: the sampler returns the injected coordinates (D=1) and runs
  # the real diffusion module once, with exactly the arguments the stock
  # sampler passes it (inference_sampler.py:173).
  def sample_diffusion_like_af3(self, *, S_inputs_I, S_trunk_I, Z_trunk_II, f,
                                diffusion_module, diffusion_batch_size,
                                coord_atom_lvl_to_be_noised):
    dev = S_inputs_I.device
    pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)[None]
    noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32,
                         device=dev)[None]
    t = torch.tensor(sigma, dtype=torch.float32, device=dev)
    got['denoised'] = diffusion_module(
        X_noisy_L=noisy, t=t.tile(1), f=f, S_inputs_I=S_inputs_I,
        S_trunk_I=S_trunk_I, Z_trunk_II=Z_trunk_II).detach().clone()
    return {'X_L': pos, 'X_noisy_L_traj': [noisy], 'X_denoised_L_traj':
            [got['denoised']], 't_hats': [t]}
  smod.SampleDiffusion.sample_diffusion_like_af3 = sample_diffusion_like_af3

  # The confidence head's logits -> pLDDT/PAE/PDE through RF3's OWN unbinning
  # (the call engine.run makes); captured before run() rounds them for JSON.
  orig_compile = emod.compile_af3_style_confidence_outputs

  def compile_conf(**kw):
    out = orig_compile(**kw)
    got['conf'] = {k: out[k].detach().clone() for k in ('plddt', 'pae', 'pde')}
    got['is_real_atom'] = kw['is_real_atom'].detach().clone()
    got['plddt_logits'] = kw['plddt_logits'].detach().clone()
    got['pae_logits'] = kw['pae_logits'].detach().clone()
    got['pde_logits'] = kw['pde_logits'].detach().clone()
    return out
  emod.compile_af3_style_confidence_outputs = compile_conf

  def patch(engine):
    ema = engine.trainer.state['model']
    # The net forward() actually runs: EMA.forward hands an eval-mode call to
    # the shadow (foundry/training/EMA.py).
    net = getattr(ema, 'module', ema)
    net = getattr(net, 'shadow', net)
    got['net'] = type(net).__name__
    orig_trunk = net.trunk_forward_with_recycling

    def trunk(*a, **kw):
      # CLONED at every yield: the recycler updates S_I/Z_II in place
      # ("We alter the S_I and Z_II in place", RF3.py), and on CPU a
      # `.cpu().numpy()` is a view; the LAST pass is the trunk output.
      for o in orig_trunk(*a, **kw):
        got['s_inputs'] = o['S_inputs_I'].detach().clone()
        got['s'] = o['S_I'].detach().clone()
        got['z'] = o['Z_II'].detach().clone()
        yield o
    net.trunk_forward_with_recycling = trunk
    net.distogram_head.register_forward_hook(
        lambda _m, _i, o: got.setdefault('distogram_logits',
                                         o.detach().clone()))
    calls = {}

    def hook(name):
      def fn(_m, _i, o):
        n = calls[name] = calls.get(name, -1) + 1
        tag = name if n == 0 else '%s@%d' % (name, n)
        for k, x in enumerate(o if isinstance(o, (tuple, list)) else (o,)):
          if hasattr(x, 'float'):
            taps['tap.%s.%d' % (tag, k)] = x.detach().float().cpu().numpy().copy()
      return fn
    for name, m in net.named_children():
      if name not in ('inference_sampler', 'mini_rollout_sampler'):
        m.register_forward_hook(hook(name))
    for name, m in net.feature_initializer.named_children():
      m.register_forward_hook(hook('feature_initializer.' + name))
    if os.environ.get('RF3_CONF_TAPS'):
      # the confidence head's own stages: the distance one-hot, the pair and
      # single entering its pairformer, each pairformer block, the heads
      ch = net.confidence_head
      for name, m in ch.named_children():
        if name != 'pairformer':
          m.register_forward_hook(hook('confidence_head.' + name))
      for i, blk in enumerate(ch.pairformer):
        blk.register_forward_hook(hook('confidence_head.pairformer.%d' % i))
      def pre(tag):
        # a pre-hook's return value REPLACES the module's input: return None
        def fn(_m, args):
          for k, x in enumerate(args):
            if hasattr(x, 'float'):
              taps.setdefault('tap.confidence_head.%s.%d' % (tag, k),
                              x.detach().float().cpu().numpy().copy())
        return fn
      ch.process_pred_distances.register_forward_pre_hook(pre('dist_one_hot'))
      ch.pairformer[0].register_forward_pre_hook(pre('pairformer_in'))

  def on_pipeline(ex):
    ex_box['ex'] = ex
    # The ref_pos the harness injected into ours must be THIS run's draw.
    now = ex['feats']['ref_pos'].detach().cpu().numpy()
    if not np.array_equal(now, lay['feat.ref_pos']):
      raise RuntimeError('rf3 re-featurised a different reference conformer '
                         'draw (max |d| %.3g): the seeded order moved'
                         % np.abs(now - lay['feat.ref_pos']).max())
    got['msa_stack_shape'] = np.array(ex['feats']['msa_stack'].shape)

  _run(out_dir, int(z['recycling_steps']) + 1, precision, on_pipeline, patch)

  # Per-atom pLDDT in the atom_array order, exactly as rf3 writes it
  # (annotate_atom_array_b_factor_with_plddt: plddt[:, is_real_atom[:, :NHEAVY]]).
  from rf3.chemical import NHEAVY
  c = got['conf']
  pl = c['plddt'][0][got['is_real_atom'][..., :NHEAVY]]
  assert pl.shape[0] == lay['atom_name'].shape[0], (pl.shape, lay['atom_name'].shape)
  out = {'s_inputs': got['s_inputs'], 's': got['s'], 'z': got['z'],
         'distogram_logits': got['distogram_logits'],
         'denoised': got['denoised'], 'pae': c['pae'][0], 'pde': c['pde'][0],
         'plddt': pl, 'pae_logits': got['pae_logits'],
         'pde_logits': got['pde_logits'], 'plddt_logits': got['plddt_logits'],
         'msa_stack_shape': got['msa_stack_shape']}
  out.update(taps)
  np.savez(out_npz, **{k: v.float().cpu().numpy().squeeze()
                       if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in out.items()})


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward}[cmd](*rest)
