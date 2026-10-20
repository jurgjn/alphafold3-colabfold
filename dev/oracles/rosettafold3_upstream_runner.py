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
(FeaturizeMSALikeAF3, n_msa=1024) is a draw too: below 1024 rows a row
permutation, which the MSA module is invariant to up to summation order; above
it (deep_msa) a random subset, lifted to all rows under MSA_ALL_ROWS.
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



def _vendor_na(msg):
  """The stock vendor cannot express this input: exit 3, graded N/A by the harness."""
  print('VENDOR_NA: ' + msg, file=sys.stderr)
  sys.exit(3)

def _strip_trailing_insertions(a3m):
  """An a3m with each row's insertions past its last aligned column dropped.

  atomworks' parse_a3m records an insertion run at the index of the column it precedes, so a run after the LAST
  column lands at index L and raises (paired_msa: "index 76 is out of bounds for axis 0 with size 76"): stock RF3
  cannot read such a row at all. AF3's parser (and ours) gives an insertion run to the next aligned column, so a
  trailing run counts for nothing there -- dropping it changes no feature on our side and lets the vendor read the
  row as we do."""
  out = []
  for line in a3m.splitlines():
    out.append(line if line.startswith('>') else line.rstrip('abcdefghijklmnopqrstuvwxyz'))
  return '\n'.join(out) + '\n'


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
      CLI reads one. (A pairedMsa goes into it, tagged by TaxID.)
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
        # rf3 templates from its INPUT structure (template_selection marks the
        # input's atoms as the template): handled by _template_cif, which
        # replaces this JSON input with a CIF of the query on the template's
        # coordinates
        _TEMPLATE_CASE.append((ids, e['sequence'], e['templates']))
      paired = e.get('pairedMsa')
      if not _trivial_msa(paired, e['sequence']):
        # rf3 reads ONE a3m per chain and pairs rows across chains by the
        # `TaxID=N` in the header (atomworks extract_tax_id): the pair a3m's
        # rows tagged TaxID=<row index>, then the single a3m's rows past its
        # query, untagged
        pr = [l for l in paired.splitlines() if l.strip()]
        ur = [l for l in (e.get('unpairedMsa') or '').splitlines() if l.strip()]
        lines = pr[:2]
        for r in range(1, len(pr) // 2):
          lines += ['>%s TaxID=%d' % (pr[2 * r][1:], 900000 + r), pr[2 * r + 1]]
        lines += ur[2:]
        p = os.path.join(msa_dir, '%s.a3m' % ids[0])
        with open(p, 'w') as f:
          f.write(_strip_trailing_insertions('\n'.join(lines) + '\n'))
        d['msa_path'] = p
      elif not _trivial_msa(e.get('unpairedMsa'), e['sequence']):
        p = os.path.join(msa_dir, '%s.a3m' % ids[0])
        with open(p, 'w') as f:
          f.write(_strip_trailing_insertions(e['unpairedMsa']))
        d['msa_path'] = p
      comps.append(d)
    elif kind == 'ligand':
      d = {'chain_id': ids if len(ids) > 1 else ids[0]}
      if 'smiles' in e:
        d['smiles'] = e['smiles']
        names = ['L:0']
      else:
        if len(e['ccdCodes']) != 1:
          _vendor_na('a multi-component CCD ligand: the rf3 JSON takes one ccd_code per chain')
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


_TEMPLATE_CASE = []


def _template_cif(j, out_dir):
  """The case's template as rf3 takes one: the query as a CIF on the template's
  coordinates, with template_selection over that chain. Only a single protein
  chain with one template maps onto this (otherwise N/A). A residue the template
  carries modified where the query has the standard one (5K9P's SEP20 under
  ubiquitin's S20) is written as the query's residue with the extra atoms gone,
  which is also what our template featurisation reads (the parent restype)."""
  import io
  import biotite.structure as bs
  from biotite.structure.io import pdbx
  polymers = [e for entry in j['sequences'] for kind, e in entry.items() if kind != 'ligand']
  if len(_TEMPLATE_CASE) != 1 or len(polymers) != 1 or len(_TEMPLATE_CASE[0][2]) != 1:
    print('VENDOR_NA: rf3 templates come from its input structure; only one templated protein chain maps',
          file=sys.stderr)
    sys.exit(3)
  ids, seq, (t,) = _TEMPLATE_CASE[0]
  # biotite, which atomworks already uses (the vendor venv has no gemmi)
  cif = pdbx.CIFFile.read(io.StringIO(t['mmcif']))
  tchain = str(cif.block['entity_poly']['pdbx_strand_id'].as_array()[0]).split(',')[0].strip()
  arr = pdbx.get_structure(cif, model=1, use_author_fields=True, extra_fields=['charge'])
  arr = arr[(arr.chain_id == tchain) & ~arr.hetero | ((arr.chain_id == tchain) & (arr.res_name == 'SEP'))]
  arr = arr[bs.filter_amino_acids(arr) | (arr.res_name == 'SEP')]
  arr = arr[arr.element != 'H']
  starts = bs.get_residue_starts(arr)
  q, tt = t['queryIndices'], t['templateIndices']
  three = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',
           'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
           'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
  drop = np.zeros(len(arr), bool)
  for qi, ti in zip(q, tt):
    lo = starts[ti]
    hi = starts[ti + 1] if ti + 1 < len(starts) else len(arr)
    want = three[seq[qi]]
    if arr.res_name[lo] != want:
      arr.res_name[lo:hi] = want
      arr.hetero[lo:hi] = False
      # the modified residue's extra atoms (SEP's phosphate) are not the query's
      drop[lo:hi] |= np.isin(arr.atom_name[lo:hi], ['P', 'O1P', 'O2P', 'O3P'])
  arr = arr[~drop]
  arr.chain_id[:] = ids[0]
  out = pdbx.CIFFile()
  pdbx.set_structure(out, arr)
  path = os.path.join(out_dir, 'rf3_template_input.cif')
  out.write(path)
  return path, [ids[0]]


def _run(out_dir, n_recycles, precision, on_pipeline, patch=None):
  """Construct, seed, build, then engine.run -- the CLI's order. on_pipeline
  sees the featurised example; patch (forward only) installs the overrides."""
  _precision()
  engine = _engine(n_recycles, precision)
  if os.environ.get('MSA_ALL_ROWS'):
    # rf3 draws n_msa (datasets n_msa: 1024) rows at random per recycle (FeaturizeMSALikeAF3,
    # uniformly_select_rows): lifted, as every side is under MSA_ALL_ROWS -- all rows, as ours
    engine.transform_overrides = dict(engine.transform_overrides, n_msa=1 << 20)
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
    tpl = os.path.join(out_dir, 'rf3_template_input.cif')
    if os.path.exists(tpl):
      with open(os.path.join(out_dir, 'template_selection.json')) as f:
        sel = json.load(f)
      engine.run(inputs=tpl, out_dir=None, template_selection=sel)
    else:
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
             f['ref_element'].detach().cpu().numpy().copy()},
      # every other per-atom feature, for an atom-by-atom diff against ours
      **{'atomfeat.' + k: v.detach().cpu().numpy().copy()
         for k, v in f.items()
         if hasattr(v, 'shape') and v.ndim >= 1 and v.shape[0] == len(aa)
         and k not in ('ref_pos', 'ref_space_uid', 'ref_element')},
      # the MSA features as atomworks built them (rows, order, pairing), for a row-by-row diff against ours
      **{'msafeat.' + k: v.detach().cpu().numpy().copy()
         for k, v in f.items()
         if hasattr(v, 'shape') and any(t in k for t in ('msa', 'profile', 'deletion', 'ins'))})


def featurise(model, case_json, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  msa_dir = os.path.join(out_dir, 'msa')
  os.makedirs(msa_dir, exist_ok=True)
  _TEMPLATE_CASE.clear()
  with open(os.path.join(out_dir, 'rf3.json'), 'w') as f:
    json.dump(_rf3_json(j, msa_dir), f, indent=1)
  if _TEMPLATE_CASE:
    path, sel = _template_cif(j, out_dir)
    with open(os.path.join(out_dir, 'template_selection.json'), 'w') as f:
      json.dump(sel, f)

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


def trajectory(model, out_dir, in_npz, out_npz):
  """RF3's REAL sampler at `rf3 fold`'s engine defaults (num_steps 50: a
  50-point schedule, 49 steps), one sample, every random draw the harness's
  (vendor_native_parity --trajectory): the initial positions, and per step
  the augmentation (foundry centre_random_augmentation, which rotates as
  R @ x -- the replacement applies the harness's x @ R) and the noise.

  RF3's initial structure is c0 * N(0, 1) PLUS coord_atom_lvl_to_be_noised,
  which inference fills with a second unit normal (extra_xforms.
  CheckForNaNsInInputs). A 1 A jitter on a 2560 A cloud, and not a draw the
  port has: it is taken as 0 here, so x0 is c0 * the harness's draw."""
  import torch
  from rf3.diffusion_samplers import inference_sampler as smod
  z = np.load(in_npz)
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32)
  n = {'eps': 0, 'aug': 0}
  got = {}

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def normal(self, mean=0.0, std=1.0, size=None, device=None, **kw):
      k = n['eps']
      n['eps'] += 1
      return eps[k].reshape(size).to(device)

  def aug(X_L, X_exists_L, s_trans=1.0):
    k = n['aug']
    n['aug'] += 1
    assert bool(X_exists_L.all())
    x = X_L - X_L.mean(dim=-2, keepdim=True)
    return x @ rot[k].to(X_L) + s_trans * trans[k].to(X_L)

  def initial(self, c0, D, L, coord_atom_lvl_to_be_noised):
    return c0 * x0.reshape(D, L, 3).to(c0.device)

  smod.torch, smod.centre_random_augmentation = _Torch(), aug
  smod.SampleDiffusion._get_initial_structure = initial
  orig = smod.SampleDiffusion.sample_diffusion_like_af3

  def sample(self, **kw):
    got['sigmas'] = self._construct_inference_noise_schedule(
        device=kw['S_inputs_I'].device).float().cpu().numpy()
    out = orig(self, **kw)
    got.setdefault('final', out['X_L'].detach().float().cpu().numpy().reshape(-1, 3))
    return out
  smod.SampleDiffusion.sample_diffusion_like_af3 = sample
  _run(out_dir, int(z['recycling_steps']) + 1, 'fp32', lambda ex: None)
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
