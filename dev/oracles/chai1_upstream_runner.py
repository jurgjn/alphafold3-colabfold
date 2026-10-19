"""The STOCK chai-1 half of vendor_native_parity.py (chai1).

  python chai1_upstream_runner.py featurise chai1 CASE.json OUT_DIR
  python chai1_upstream_runner.py forward   chai1 OUT_DIR IN.npz OUT.npz

Run in ~/venv_chai (chai_lab 0.6.1 from PyPI, nothing else on the path), with
CHAI_DOWNLOADS_DIR naming the downloaded components. Everything is chai's own,
driven the way `chai_lab.chai1.run_inference` drives it:
make_all_atom_feature_context -> run_folding_on_context, which collates the
features and calls the exported TorchScript components in sequence.

chai ships TorchScript, so nothing INSIDE a component can be hooked; what
crosses BETWEEN them is recorded by wrapping `load_exported`. The sampler is
inline in run_folding_on_context, so instead of overriding it:
  * the diffusion wrapper's first call runs the real module ONCE on the
    injected (noised coordinates, sigma) and records it -- what it returns to
    chai's loop is irrelevant, the loop is cut to one step;
  * the confidence wrapper scores the injected coordinates.

INPUTS. chai's FASTA takes polymers by sequence, a modified residue inline as
`(CCD)`, and a ligand ONLY as SMILES (its residue is always `LIG`): a CCD
ligand goes in as that CCD entry's SMILES, so its atom names are chai's, and
the harness aligns those atoms by token + element. A glycan goes in as chai's
glycan entity, a covalent bond as a restraints-CSV `covalent` row. An MSA is
written as chai's own ColabFold path writes one (colabfold.py): the query as
the paired block, the unpaired rows not already in it, a source per header.
ESM embeddings are off (use_esm_embeddings=False: zeros), as on our side
without ESM_EMB.

There is no distogram: stock chai-lab ships no distogram head.
"""
import json
import os
import sys

import numpy as np

# SMILES for the CCD ligands the cases use (chai takes none by code).
_CCD_SMILES = {
    'ATP': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O'
           '[P@](=O)(O)OP(=O)(O)O)O)O)N',
    'MG': '[Mg+2]',
}


class _Stop(Exception):
  pass


def _case_inputs(j, d):
  """AF3 fold-input JSON -> (fasta path, msa dir or None, restraints path or None)."""
  from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
  import pandas as pd
  lines, restraints, msa_dir = [], [], None
  glycan_chains = set()
  for pair in j.get('bondedAtomPairs') or ():
    for a in pair:
      for entry in j['sequences']:
        (kind, e), = entry.items()
        if kind == 'ligand' and a[0] in (e['id'] if isinstance(e['id'], list) else [e['id']]):
          glycan_chains.add(a[0])
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    for cid in ids:
      if kind in ('protein', 'rna', 'dna'):
        seq = list(e['sequence'])
        for m in e.get('modifications', []):
          pos = m.get('ptmPosition', m.get('basePosition'))
          seq[pos - 1] = '(%s)' % m.get('ptmType', m.get('modificationType'))
        lines += ['>%s|name=%s' % (kind, cid), ''.join(seq)]
        if kind == 'protein' and e.get('unpairedMsa') and len(e['unpairedMsa'].splitlines()) > 2:
          msa_dir = msa_dir or os.path.join(d, 'msa')
          os.makedirs(msa_dir, exist_ok=True)
          path = os.path.join(msa_dir, expected_basename(e['sequence']))
          if not os.path.exists(path):
            _write_pqt(e['sequence'], e['unpairedMsa'], path, pd)
      elif kind == 'ligand':
        if cid in glycan_chains:
          codes = e['ccdCodes']
          lines += ['>glycan|name=%s' % cid, ''.join('%s' % c for c in codes)]
        elif 'smiles' in e:
          lines += ['>ligand|name=%s' % cid, e['smiles']]
        else:
          (code,) = e['ccdCodes']
          lines += ['>ligand|name=%s' % cid, _CCD_SMILES[code]]
  for k, (a, b) in enumerate(j.get('bondedAtomPairs') or ()):
    def ref(x):
      # chai's res_idx: residue one-letter + 1-based index, '@' atom; a ligand
      # or glycan names only its atom
      for entry in j['sequences']:
        (kind, e), = entry.items()
        if x[0] in (e['id'] if isinstance(e['id'], list) else [e['id']]):
          if kind == 'protein':
            return '%s%d@%s' % (e['sequence'][x[1] - 1], x[1], x[2])
          return '@%s' % x[2] if kind == 'ligand' and x[0] not in glycan_chains else '%s%d@%s' % ('', x[1], x[2])
      raise KeyError(x)
    restraints.append(dict(restraint_id='bond%d' % k, chainA=a[0], res_idxA=ref(a), chainB=b[0], res_idxB=ref(b),
                           connection_type='covalent', confidence=1.0, min_distance_angstrom=0.0,
                           max_distance_angstrom=0.0, comment='bondedAtomPairs'))
  fasta = os.path.join(d, 'input.fasta')
  with open(fasta, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  rpath = None
  if restraints:
    rpath = os.path.join(d, 'restraints.csv')
    pd.DataFrame(restraints).to_csv(rpath, index=False)
  return fasta, msa_dir, rpath


def _write_pqt(query, a3m, path, pd):
  """One chain's aligned.pqt as chai's ColabFold path writes it (colabfold.py)."""
  from chai_lab.data.parsing.msas.data_source import MSADataSource
  rows, header = [], None
  for line in a3m.splitlines():
    if line.startswith('>'):
      header = line[1:]
    elif line.strip():
      rows.append((header, line.strip()))
  # paired block: the query alone (AF3's pairedMsa is empty in every case here)
  paired = [('0', rows[0][0], rows[0][1])]
  in_paired = {s for _, _, s in paired}
  single = [(h, s) for i, (h, s) in enumerate(rows) if i > 0 and s not in in_paired]
  headers = [h for _, h, _ in paired] + [h for h, _ in single]
  sources = ['query'] + [MSADataSource.UNIREF90.value if h.startswith('UniRef') else MSADataSource.BFD_UNICLUST.value
                         for h in headers[1:]]
  df = pd.DataFrame(dict(sequence=[s for _, _, s in paired] + [s for _, s in single], source_database=sources,
                         pairing_key=['0'] + [''] * len(single), comment=''))
  df.to_parquet(path)


def _np(t):
  import torch
  t = t.detach()
  if t.dtype == torch.bfloat16:
    t = t.float()
  return t.cpu().numpy()


def _run(case_dir, recycles, on_inputs, inject=None):
  """chai's own featurise + fold, with every exported component wrapped."""
  from pathlib import Path
  import torch
  import chai_lab.chai1 as C
  from chai_lab.utils.tensor_utils import set_seed

  with open(os.path.join(case_dir, 'case.json')) as f:
    j = json.load(f)
  fasta, msa_dir, rpath = _case_inputs(j, case_dir)
  set_seed([1])
  out = Path(case_dir) / 'chai_out'
  import shutil
  shutil.rmtree(out, ignore_errors=True)
  ctx = C.make_all_atom_feature_context(
      Path(fasta), output_dir=out, use_esm_embeddings=False,
      msa_directory=Path(msa_dir) if msa_dir else None,
      constraint_path=Path(rpath) if rpath else None)
  got = {}
  orig_collate = C.Collate

  class Collate(orig_collate):
    def __call__(self, contexts):
      batch = super().__call__(contexts)
      got['inputs'] = batch['inputs']
      if os.environ.get('CHAI_DUMP_FEATURES'):
        # chai's raw (pre-embedding) feature tensors, for a field-by-field diff
        np.savez(os.path.join(case_dir, 'chai_features.npz'),
                 **{k: _np(v)[0] for k, v in batch['features'].items()
                    if hasattr(v, 'shape') and v.numel() < 5e7})
      on_inputs(batch['inputs'])
      return batch
  C.Collate = Collate

  class Wrap:
    def __init__(self, name, inner):
      self.name, self.inner, self.calls = name, inner, 0

    def forward(self, **kw):
      self.calls += 1
      if self.name == 'diffusion_module.pt' and self.calls == 1:
        dev = kw['atom_noised_coords'].device
        kw = dict(kw, atom_noised_coords=torch.tensor(inject['noisy'], device=dev)[None, None],
                  noise_sigma=torch.full((1, 1), inject['sigma'], device=dev))
        out = self.inner.forward(**kw)
        got['denoised'] = _np(out).reshape(-1, 3)   # (1, [ds,] atoms, 3) -> (atoms, 3)
        return out
      if self.name == 'confidence_head.pt':
        kw = dict(kw, atom_coords=torch.tensor(inject['positions'], device=kw['atom_coords'].device)[None])
      out = self.inner.forward(**kw)
      if self.name == 'feature_embedding.pt':
        got['emb.TOKEN'] = _np(out['TOKEN'])[0]
        got['emb.ATOM'] = _np(out['ATOM'])[0]
      if self.name == 'token_embedder.pt':
        got['s_inputs'] = _np(out[0])[0]
      elif self.name == 'trunk.pt':
        got['s'], got['z'] = _np(out[0])[0], _np(out[1])[0]
      elif self.name == 'confidence_head.pt':
        got['pae_logits'], got['pde_logits'], got['plddt_logits'] = (_np(x)[0] for x in out)
      return out

  orig = C.load_exported
  C.load_exported = lambda key, device: Wrap(key, orig(key, device))
  try:
    C.run_folding_on_context(ctx, output_dir=out, num_trunk_recycles=recycles + 1, num_diffn_timesteps=2,
                             num_diffn_samples=1, seed=1, device=torch.device('cuda:0'), low_memory=False)
  except _Stop:
    pass
  finally:
    C.load_exported, C.Collate = orig, orig_collate
  return got


def _layout(inputs):
  real = _np(inputs['atom_exists_mask'])[0].astype(bool)
  names = [n for n, r in zip(inputs['atom_ref_name'][0] if isinstance(inputs['atom_ref_name'][0], list)
                             else inputs['atom_ref_name'], real) if r]
  return dict(
      atom_name=np.array([str(n) for n in names]),
      atom_token=_np(inputs['atom_token_index'])[0][real],
      atom_pad=np.ones(int(real.sum()), bool),
      token_pad=_np(inputs['token_exists_mask'])[0].astype(bool)[_np(inputs['token_exists_mask'])[0].astype(bool)],
      atom_slot=np.flatnonzero(real),
      **{'feat.ref_pos': _np(inputs['atom_ref_pos'])[0][real],
         'feat.ref_space_uid': _np(inputs['atom_ref_space_uid'])[0][real],
         'feat.ref_element': np.eye(128, dtype=np.int64)[_np(inputs['atom_ref_element'])[0][real]],
         'atomfeat.ref_charge': _np(inputs['atom_ref_charge'])[0][real],
         'atomfeat.ref_name_chars': _np(inputs['atom_ref_name_chars'])[0][real]})


def featurise(model, case_json, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  import shutil
  shutil.copy(case_json, os.path.join(out_dir, 'case.json'))

  def on_inputs(inputs):
    np.savez(os.path.join(out_dir, 'layout.npz'), **_layout(inputs))
    raise _Stop
  _run(out_dir, 0, on_inputs)


def forward(model, out_dir, in_npz, out_npz):
  z = np.load(in_npz)
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  slot = lay['atom_slot']

  def padded(x):
    out = np.zeros((5888 if slot.max() < 5888 else int(slot.max()) + 1, 3), np.float32)
    out[slot] = x
    return out
  state = {}

  def on_inputs(inputs):
    n_atoms = inputs['atom_exists_mask'].shape[-1]
    state['n'] = n_atoms
    now = _np(inputs['atom_ref_pos'])[0][slot]
    if not np.array_equal(now, lay['feat.ref_pos']):
      raise RuntimeError('chai re-drew a different reference conformer (max |d| %.3g)'
                         % np.abs(now - lay['feat.ref_pos']).max())

  def pad_to(x, n):
    out = np.zeros((n, 3), np.float32)
    out[slot] = x
    return out
  # the padded atom count is only known after collation; chai pads to 23 * the token bucket
  n_tok_bucket = None
  inject = {'sigma': float(z['noise_level'])}

  class Lazy(dict):
    def __getitem__(self, k):
      n = state['n']
      if k == 'noisy':
        return pad_to(z['positions_noisy'], n)
      if k == 'positions':
        return pad_to(z['positions'], n)
      return dict.__getitem__(self, k)
  got = _run(out_dir, int(z['recycling_steps']), on_inputs, inject=Lazy(inject))
  inputs = got['inputs']
  tok = _np(inputs['token_exists_mask'])[0].astype(bool)
  n = int(tok.sum())
  centres = lambda lo, hi, k: (np.linspace(lo, hi, k + 1)[:-1] + np.linspace(lo, hi, k + 1)[1:]) / 2

  def expect(logits, c):
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True) * c).sum(-1)
  pl_atoms = expect(got['plddt_logits'], centres(0, 1, got['plddt_logits'].shape[-1]))
  np.savez(out_npz,
           s_inputs=got['s_inputs'][:n], s=got['s'][:n], z=got['z'][:n, :n],
           pae=expect(got['pae_logits'][:n, :n], centres(0, 32, 64)),
           pde=expect(got['pde_logits'][:n, :n], centres(0, 32, 64)),
           plddt=pl_atoms[slot], denoised=got['denoised'][slot],
           **{'tap.emb_token': got['emb.TOKEN'][:n], 'tap.emb_atom': got['emb.ATOM'][slot]})


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward}[cmd](*rest)
