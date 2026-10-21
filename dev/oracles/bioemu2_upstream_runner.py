"""Stock BioEmu2 (microsoft/bioemu2 @7846769) for vendor_native_parity.

    python bioemu2_upstream_runner.py featurise MODEL CASE.json DIR
    python bioemu2_upstream_runner.py forward MODEL DIR IN.npz OUT.npz
    python bioemu2_upstream_runner.py trajectory MODEL DIR IN.npz OUT.npz

Runs in ~/venv_bioemu2 (`uv sync --frozen --extra cuda`), against the checkpoint
in ~/bioemu2_ckpt (BIOEMU2_CKPT), in float32 at 'highest' matmul precision:
`load_model(precision='32-true', matmul_precision=None)` -- the stock default is
a bf16 trunk and a TF32 denoiser, which is a precision choice, not the model.

An AF3 fold input becomes a BioEmu2 System by the rule
pipeline/bioemu2_features.py documents (and ours follows): identical chains are
one entity, in order of first appearance; a one-code ligand with no declared
bond is a Ligand, one with several codes or any bond is a Branched entity; a
SMILES ligand is a Ligand from SMILES. MSAs are the JSON's own A3Ms; an empty
one is the query alone; paired A3Ms are handed over when every protein chain
has one.

featurise writes DIR/feats.npz: the vendor's model feature dict
(convert_densebatch_to_dictionary, batch axis squeezed), plus `smiles.<chain>.*`
for every SMILES ligand -- that conformer is an unseeded RDKit draw, so the
harness hands it to our side.
"""
import os as _os, sys as _sys  # dev/af3_dev_paths.py: vendor trees, venvs and weights
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from af3_dev_paths import dev_path  # noqa: E402
import json
import os
import sys

import numpy as np

H = dev_path('~')
CKPT = os.environ.get('BIOEMU2_CKPT', H + '/bioemu2_ckpt/checkpoint.ckpt')
THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
         'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
         'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}


def _strict():
  import torch
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')


def _chains(j):
  """[(kind, chain id, entry)] in JSON order, ids expanded."""
  out = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    for cid in (e['id'] if isinstance(e['id'], list) else [e['id']]):
      out.append((kind, cid, e))
  return out


def _msa(e, key, base):
  """An inline MSA, or the file its ...Path names (relative to the JSON), as AF3 reads it."""
  if e.get(key):
    return e[key]
  path = e.get(key + 'Path')
  if path:
    with open(os.path.join(base, path)) as fh:
      return fh.read()
  return None


def system_and_msa(j, base='.'):
  from bioemu2.data_interface import System
  from bioemu2.data_interface.structure_hierarchy import (
      Atom, Bond, Branched, DNA, Ligand, ModifiedResidue, Protein, RNA)
  from bioemu2.msa.alignment import MSA
  bonds = j.get('bondedAtomPairs') or []
  bonded = {a[0] for pair in bonds for a in pair}
  groups, order = {}, []
  for kind, cid, e in _chains(j):
    if kind in ('protein', 'rna', 'dna'):
      mods = tuple(sorted((m.get('ptmType') or m.get('modificationType'),
                           m.get('ptmPosition') or m.get('basePosition'))
                          for m in e.get('modifications', [])))
      key = (kind, e['sequence'], mods)
    elif e.get('smiles'):
      key = ('smiles', e['smiles'], ())
    elif len(e['ccdCodes']) == 1 and cid not in bonded:
      key = ('ccd', e['ccdCodes'][0], ())
    else:
      key = ('branched', tuple(e['ccdCodes']), ())
    if key not in groups:
      groups[key] = []
      order.append(key)
    groups[key].append((cid, e))
  entities, res_name = [], {}
  unpaired, paired = {}, {}
  for key in order:
    kind, payload, mods = key
    ids = [cid for cid, _ in groups[key]]
    if kind in ('protein', 'rna', 'dna'):
      cls = {'protein': Protein, 'rna': RNA, 'dna': DNA}[kind]
      entities.append(cls(sequence=payload, asym_ids=ids,
                          mods=[ModifiedResidue(res_id=p, ccd_code=c) for c, p in mods]))
      for cid in ids:
        names = {}
        for i, letter in enumerate(payload, start=1):
          names[i] = (THREE.get(letter, 'UNK') if kind == 'protein'
                      else (letter if kind == 'rna' else 'D' + letter))
        for c, p in mods:
          names[p] = c
        res_name[cid] = names
      if kind == 'protein':
        for cid, e in groups[key]:
          unpaired[cid] = _msa(e, 'unpairedMsa', base) or '>query\n%s\n' % payload
          paired[cid] = _msa(e, 'pairedMsa', base) or ''
    elif kind == 'smiles':
      entities.append(Ligand(smiles=payload, asym_ids=ids))
    elif kind == 'ccd':
      entities.append(Ligand(ccd_code=payload, asym_ids=ids))
      for cid in ids:
        res_name[cid] = {1: payload}
    else:
      entities.append(Branched(mon_seq=[ModifiedResidue(ccd_code=c) for c in payload], asym_ids=ids))
      for cid in ids:
        res_name[cid] = {i: c for i, c in enumerate(payload, start=1)}
  sys_bonds = [Bond(atom1=Atom(name=a[2], res_name=res_name[a[0]][a[1]], res_id=a[1], asym_id=a[0]),
                    atom2=Atom(name=b[2], res_name=res_name[b[0]][b[1]], res_id=b[1], asym_id=b[0]))
               for a, b in bonds]
  system = System(entities, bonds=sys_bonds) if sys_bonds else System(entities)
  seqs = {cid: e['sequence'] for kind, cid, e in _chains(j) if kind == 'protein'}
  hetero = len(set(seqs.values())) > 1
  use_paired = hetero and all(paired[c] for c in unpaired)
  msa = MSA(unpaired_a3m=unpaired, paired_a3m=paired if use_paired else None)
  return system, msa


def featurise(model, case_json, out_dir):
  import torch
  from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
  from bioemu2.structure.batch import batch_collate
  from bioemu2.structure.preparation import prepare_structure
  from bioemu2.structure import decode_atom_name
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as fh:
    j = json.load(fh)
  system, msa = system_and_msa(j, os.path.dirname(os.path.abspath(case_json)))
  with open(os.path.join(out_dir, 'case.json'), 'w') as fh:
    json.dump(j, fh)
  mode = os.environ.get('BIOEMU2_MODE', 'structure')
  contacts = []
  if os.environ.get('BIOEMU2_CONTACTS'):
    from bioemu2.data_interface.structure_hierarchy import Contact
    with open(os.environ['BIOEMU2_CONTACTS']) as fh:
      contacts = [Contact(**c) for c in json.load(fh)]
  st = prepare_structure(system, msa=msa, mode=mode, contacts=contacts)
  batch = batch_collate([st])
  torch.save(batch, os.path.join(out_dir, 'batch.pt'))
  feats = convert_densebatch_to_dictionary(batch)
  out = {k: v[0].cpu().numpy() for k, v in feats.items() if torch.is_tensor(v)}
  out['dt'] = out['dt'].reshape(())
  out['is_dynamics_conditioned'] = out['is_dynamics_conditioned'].reshape(())
  # the SMILES ligands' conformers (an unseeded draw): per chain, in the vendor's order
  asym = st.allatom_asym_id.numpy()
  chain_ids = system.chain_ids
  bi = st.bond_index.numpy()
  for e in system.entities:
    if getattr(e, 'smiles', None):
      for cid in e.asym_ids:
        sel = np.flatnonzero(asym == chain_ids.index(cid))
        lo = sel[0]
        keep = np.isin(bi[0], sel) & np.isin(bi[1], sel)
        out['smiles.%s.coords' % cid] = st.template_coords.numpy()[sel]
        out['smiles.%s.names' % cid] = np.array(
            [decode_atom_name(x) for x in st.name_encoding.numpy()[sel].tolist()])
        out['smiles.%s.element' % cid] = st.element.numpy()[sel]
        out['smiles.%s.charge' % cid] = st.charge.numpy()[sel].astype(np.float32)
        out['smiles.%s.bonds' % cid] = bi[:, keep] - lo
  np.savez(os.path.join(out_dir, 'feats.npz'), **out)


def _load():
  from bioemu2 import load_model
  _strict()
  return load_model(checkpoint_path=CKPT, precision='32-true', matmul_precision=None)


def forward(model, out_dir, in_npz, out_npz):
  import torch
  z = np.load(in_npz)
  m = _load()
  sm = m.score_model
  batch = torch.load(os.path.join(out_dir, 'batch.pt'), weights_only=False).to(m.device)
  with torch.no_grad():
    sm_recycles = int(z['recycling_steps'])
    emb = sm.compute_trunk_embeddings(batch, sm_recycles)
    ctx = sm.compute_static_context(batch, emb)
    noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32, device=m.device)[None]
    sigma = torch.tensor([float(z['noise_level'])], device=m.device)
    den = sm.denoise(ctx, emb, noisy, sigma).denoised_coords
  np.savez(out_npz, s_inputs=emb.s_inputs[0].cpu().numpy(), s=emb.single[0].cpu().numpy(),
           z=emb.pair[0].cpu().numpy(), distogram_logits=emb.distogram[0].cpu().numpy(),
           pair_cond=ctx.z[0].cpu().numpy(), denoised=den[0].cpu().numpy())


def _centred(coords, batch):
  """sampling.py's last step: each sample centred on its real atoms (they lead)."""
  n_real = int(batch.atom_mask[0].sum())
  c = coords.clone()
  c[:, :n_real] = c[:, :n_real] - c[:, :n_real].mean(dim=1, keepdim=True)
  return c


def trajectory(model, out_dir, in_npz, out_npz):
  """BioEmu2's EDMSampler at its defaults for the mode (200 steps, step_scale
  1.5 structure / 1.0 equilibrium), every draw the harness's: the initial
  standard-normal positions, and per step the rotation, translation and eps."""
  import torch
  from bioemu2.denoising import edm as E
  from bioemu2.denoising.samplers import EDMSampler
  z = np.load(in_npz)
  m = _load()
  batch = torch.load(os.path.join(out_dir, 'batch.pt'), weights_only=False).to(m.device)
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32, device=m.device)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32, device=m.device)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32, device=m.device)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32, device=m.device)
  n = {'randn': 0, 'aug': 0}

  class _Sync:
    def __getattr__(self, k):
      return getattr(E._orig_cp_sync, k)

    def randn(self, shape, device=None, dtype=None):
      k = n['randn']
      n['randn'] += 1
      v = x0 if k == 0 else eps[k - 1]
      return v.reshape(shape).to(dtype or torch.float32)

    def random_augmentation(self, multiplicity, device=None, dtype=torch.float32):
      k = n['aug']
      n['aug'] += 1
      return rot[k][None].to(dtype), trans[k].reshape(1, 1, 3).to(dtype)

  E._orig_cp_sync = E.cp_sync
  E.cp_sync = _Sync()
  mode = os.environ.get('BIOEMU2_MODE', 'structure')
  sampler = EDMSampler(step_scale=1.5 if mode == 'structure' else 1.0)
  with torch.no_grad():
    # the harness's recycle count, as `forward` -- not load_model's default 3
    emb = m.score_model.compute_trunk_embeddings(batch, int(z['recycling_steps']))
    if 'x0_frame' in z:
      # propagate_with_embeddings, for one frame given in angstrom: coords_cond
      # the frame centred, the flag, dt in ps; then the endpoint superposed on it
      from bioemu2.denoising.rigid_align import weighted_rigid_align
      x = torch.tensor(z['x0_frame'], dtype=torch.float32, device=m.device)[None]
      cond = x.new_zeros(1, batch.atom_pad_mask.shape[1], 3)
      cond[:, :x.shape[1]] = x - x.mean(dim=1, keepdim=True)
      batch = batch.replace(coords_cond=cond,
                            is_dynamics_conditioned=torch.ones((1, 1), dtype=torch.bool, device=m.device),
                            dt=torch.full((1, 1), float(z['lag_ns']) * 1000.0, device=m.device))
    out = sampler(batch=batch, score_model=m, trunk_embeddings=emb, batch_size=1)
    if 'x0_frame' in z:
      n_real = x.shape[1]
      end = out.coords[:, :n_real].float()
      w = torch.ones((1, n_real), device=m.device)
      end, _, _ = weighted_rigid_align(end, x, w, w)
      out = out.replace(coords=end)
    else:
      out = out.replace(coords=_centred(out.coords, batch))
  sched = m.edm_schedule
  sig = E.edm_sample_schedule(sampler.num_steps, sigma_min=sched.sigma_min,
                              sigma_max=sched.sigma_max, sigma_data=sched.sigma_data,
                              device='cpu').numpy()
  np.savez(out_npz, sigmas=sig, steps=n['aug'], final=out.coords[0].float().cpu().numpy())


def steer(model, out_dir, in_npz, out_npz):
  """A steering preset (BIOEMU2_SAMPLER: edm_gd, edm_fk, edm_fkc) over P particles,
  every draw the harness's: x0 (P, N, 3); per step rot (P, 3, 3), trans (P, 3),
  eps (P, N, 3); and the resampling uniforms -- by call for edm_gd/edm_fk, by step
  for edm_fkc, whose resampling is adaptive. Also writes the potentials' ligand
  (RDKit) tables, which ours takes from here."""
  import torch
  from bioemu2.denoising import edm as E
  from bioemu2.denoising.steering import edm_fkc as FKC
  from bioemu2.denoising.steering import edm_steered as ST
  from bioemu2.denoising.steering import feats as F
  from bioemu2.denoising.samplers import FKCSampler, GuidedFKSampler
  from bioemu2.denoising.steering.potentials import get_default_potentials
  z = np.load(in_npz)
  m = _load()
  with open(os.path.join(out_dir, 'case.json')) as fh:
    system, _ = system_and_msa(json.load(fh))
  batch = torch.load(os.path.join(out_dir, 'batch.pt'), weights_only=False).to(m.device)
  dev = m.device
  x0 = torch.tensor(z['x0'], device=dev)
  rot, trans = torch.tensor(z['rot'], device=dev), torch.tensor(z['trans'], device=dev)
  eps, uni = torch.tensor(z['eps'], device=dev), torch.tensor(z['uniform'], device=dev)
  preset = os.environ['BIOEMU2_SAMPLER']
  n = {'randn': 0, 'aug': 0, 'rand': 0}

  def randn(shape, **kw):
    k = n['randn']
    n['randn'] += 1
    return (x0 if k == 0 else eps[k - 1]).reshape(shape).to(kw.get('dtype') or torch.float32)

  def aug(multiplicity, *a, **kw):
    k = n['aug']
    n['aug'] += 1
    return rot[k], trans[k].reshape(-1, 1, 3)

  class _Sync:
    def __getattr__(self, k):
      return getattr(E.cp_sync, k)

    def randn(self, shape, device=None, dtype=None):
      return randn(shape, dtype=dtype)

    def random_augmentation(self, multiplicity, device=None, dtype=torch.float32):
      return aug(multiplicity)

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def randn_like(self, x):
      return randn(x.shape, dtype=x.dtype)

    def rand(self, *a, **kw):
      k = n['rand']
      n['rand'] += 1
      # by call for the fixed-interval sampler; by step (the augmentation just drawn) for fkc
      return uni[n['aug'] - 1 if preset == 'edm_fkc' else k].reshape(()).to(kw.get('dtype') or torch.float32)

  ST.cp_sync = _Sync()
  trace = [] if os.environ.get('BIOEMU2_FK_TRACE') and 'floor' not in out_npz else None
  if trace is not None:
    # dev: each resampling step's weights, the indices drawn from them, and the energies
    _gs, _sr, _lg = ST.grouped_softmax, ST.systematic_resample, ST.fk_log_increment

    def _grouped(un, g):
      w = _gs(un, g)
      trace.append({'w': w.float().cpu().numpy()})
      return w

    def _resample(w, g):
      idx = _sr(w, g)
      trace[-1]['idx'] = idx.cpu().numpy()
      return idx

    def _increment(e, prev):
      trace.append({'energy': e.float().cpu().numpy()})
      return _lg(e, prev)
    ST.grouped_softmax, ST.systematic_resample, ST.fk_log_increment = _grouped, _resample, _increment
  FKC.torch = _Torch()
  FKC.compute_random_augmentation = aug
  F.torch = _Torch()
  if os.environ.get('BIOEMU2_POTENTIALS'):
    from pathlib import Path
    from bioemu2.cli import _load_potentials
    potentials = _load_potentials(preset, Path(os.environ['BIOEMU2_POTENTIALS']))
  else:
    potentials = get_default_potentials()
  for p in potentials:
    p.prepare(system)
  P = x0.shape[0]
  mode = os.environ.get('BIOEMU2_MODE', 'structure')
  if preset == 'edm_fkc':
    sampler = FKCSampler(potentials=potentials)
  else:
    sampler = GuidedFKSampler(step_scale=1.5 if mode == 'structure' else 1.0, potentials=potentials,
                              resampling_interval=None if preset == 'edm_gd' else 3)
  x_frame = None
  if 'x0_frame' in z:
    # propagate_with_embeddings for P copies of one frame (angstrom)
    x_frame = torch.tensor(z['x0_frame'], dtype=torch.float32, device=dev)[None].expand(P, -1, -1)
    cond = x_frame.new_zeros(P, batch.atom_pad_mask.shape[1], 3)
    cond[:, :x_frame.shape[1]] = x_frame - x_frame.mean(dim=1, keepdim=True)
    batch = batch.replace(coords_cond=cond,
                          is_dynamics_conditioned=torch.ones((1, P), dtype=torch.bool, device=dev),
                          dt=torch.full((1, P), float(z['lag_ns']) * 1000.0, device=dev))
  with torch.no_grad():
    emb = m.score_model.compute_trunk_embeddings(batch, int(z['recycling_steps']))
    out = sampler(batch=batch, score_model=m, trunk_embeddings=emb, batch_size=P)
    if x_frame is not None:
      from bioemu2.denoising.rigid_align import weighted_rigid_align
      n_real = x_frame.shape[1]
      w = torch.ones((P, n_real), device=dev)
      end, _, _ = weighted_rigid_align(out.coords[:, :n_real].float(), x_frame, w, w)
      out = out.replace(coords=end)
    else:
      out = out.replace(coords=_centred(out.coords, batch))
  pb = next((p.feats for p in potentials if 'rdkit_bounds_index' in getattr(p, 'feats', {})),
            None)
  if pb is None:
    tmp_pb = get_default_potentials()[2]
    tmp_pb.prepare(system)
    pb = tmp_pb.feats
  if trace:
    rows = [{**a, **b} for a, b in zip(trace[0::2], trace[1::2])]
    np.savez(os.environ['BIOEMU2_FK_TRACE'] + '.vendor.npz', w=np.stack([r['w'] for r in rows]),
             idx=np.stack([r['idx'] for r in rows]), energy=np.stack([r['energy'] for r in rows]))
  np.savez(out_npz, final=out.coords.float().cpu().numpy(), steps=n['aug'],
           **{'lig.' + k: pb[k][0].cpu().numpy() for k in (
               'rdkit_bounds_index', 'rdkit_lower_bounds', 'rdkit_upper_bounds',
               'rdkit_bounds_bond_mask', 'rdkit_bounds_angle_mask', 'chiral_atom_index',
               'chiral_atom_orientations', 'stereo_bond_index', 'stereo_bond_orientations',
               'planar_bond_index')})


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory, 'steer': steer}[cmd](*rest)
