"""BioEmu2's protein physicality check, for its output samples.

A transcription of bioemu2 @7846769's physicality.py (check_physicality and
what it calls) in numpy, on the topology pipeline/bioemu2_features.py returns
and the bond graph pipeline/bioemu2_steering.py builds (the vendor's MDTraj
topology: the same atoms and bonds). It changes no coordinates: per sample it
reports whether the protein chains pass the preset's thresholds and the first
check that failed, as BioEmu2's `physicality.json` does. Non-protein chains are
not checked, as in BioEmu2.
"""

import numpy as np

PROT_BACKBONE = ('N', 'CA', 'C', 'O')
BACKBONE_ATOMS = frozenset((*PROT_BACKBONE, 'OXT'))
MAINCHAIN_BOND_WINDOWS = {('N', 'CA'): (1.25, 1.70), ('CA', 'C'): (1.35, 1.70),
                          ('C', 'O'): (0.95, 1.50), ('C', 'OXT'): (0.95, 1.50)}
PEPTIDE_BOND_WINDOW = (1.05, 1.65)
SIDECHAIN_BOND_WINDOWS = {('C', 'C'): (1.10, 1.80), ('C', 'N'): (1.10, 1.70),
                          ('C', 'O'): (1.00, 1.60), ('C', 'S'): (1.55, 2.10),
                          ('C', 'SE'): (1.70, 2.25)}
DISULFIDE_BOND_WINDOW = (1.80, 2.25)
GENERIC_BOND_WINDOW = (1.00, 1.90)
MAX_HEAVY_NEIGHBORS = {'C': 4, 'N': 4, 'O': 2, 'S': 6, 'P': 5}
SIDECHAIN_BONDS = {
    'ALA': {('CA', 'CB')},
    'ARG': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')},
    'ASN': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')},
    'ASP': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')},
    'CYS': {('CA', 'CB'), ('CB', 'SG')},
    'GLN': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2')},
    'GLU': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')},
    'GLY': set(),
    'HIS': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2'), ('CE1', 'NE2')},
    'ILE': {('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2'), ('CG1', 'CD1')},
    'LEU': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')},
    'LYS': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')},
    'MET': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')},
    'PHE': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ')},
    'PRO': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'N')},
    'SER': {('CA', 'CB'), ('CB', 'OG')},
    'THR': {('CA', 'CB'), ('CB', 'OG1'), ('CB', 'CG2')},
    'TRP': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('NE1', 'CE2'), ('CD2', 'CE2'),
            ('CD2', 'CE3'), ('CE2', 'CZ2'), ('CE3', 'CZ3'), ('CZ2', 'CH2'), ('CZ3', 'CH2')},
    'TYR': {('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'),
            ('CE2', 'CZ'), ('CZ', 'OH')},
    'VAL': {('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2')},
}
LIGAND_RESNAMES = ('LIG', 'L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08', 'L09')
THRESHOLDS = {
    'no_violations': dict(bad_bonds_backbone=0, bad_bonds_sidechain=0, bad_bonds_sulfur=0,
                          within_residue_clashes=0, valence_exceeded=0, isolated_atoms=0,
                          same_chain_residue_clashes=0, cross_chain_residue_clashes=0,
                          duplicate_atoms=0, atoms_mismatch=0, bonds_mismatch=0, per_residue=False),
    'high_quality': dict(bad_bonds_backbone=5, bad_bonds_sidechain=10, bad_bonds_sulfur=0,
                         within_residue_clashes=5, valence_exceeded=10, isolated_atoms=1,
                         same_chain_residue_clashes=10, cross_chain_residue_clashes=5,
                         duplicate_atoms=0, atoms_mismatch=0, bonds_mismatch=0, per_residue=False),
    'high_quality_per_residue': dict(bad_bonds_backbone=0.05, bad_bonds_sidechain=0.1, bad_bonds_sulfur=0.0,
                                     within_residue_clashes=0.05, valence_exceeded=0.1, isolated_atoms=0.01,
                                     same_chain_residue_clashes=0.1, cross_chain_residue_clashes=0.05,
                                     duplicate_atoms=0.0, atoms_mismatch=0.0, bonds_mismatch=0.0,
                                     per_residue=True),
}
CUTOFFS = dict(too_close=1.0, inferred_bond=1.9, isolated_atom=2.0, sidechain_clash=2.0, degenerate=0.5)


def _symbol(z):
  from rdkit import Chem
  return Chem.GetPeriodicTable().GetElementSymbol(int(z)).upper() if z else 'X'


def _key(a, b):
  return (a, b) if a <= b else (b, a)


class _Protein:
  """The checked atoms (multi-residue chains with a standard amino acid), as the
  vendor's atom_slice leaves them: per atom its name, residue, chain, element,
  and the bonds between kept atoms."""

  def __init__(self, topo, bonds, ccd):
    names, labels = topo['names'], topo['labels']
    res, chain = topo['residue_idx'], topo['atom_asym']
    chains = {}
    for i in range(len(names)):
      chains.setdefault(int(chain[i]), set()).add(int(res[i]))
    keep_chain = {c for c, rs in chains.items() if len(rs) > 1 and any(
        labels[i] in SIDECHAIN_BONDS for i in range(len(names)) if chain[i] == c)}
    self.idx = np.array([i for i in range(len(names)) if int(chain[i]) in keep_chain], np.int64)
    where = {int(o): k for k, o in enumerate(self.idx)}
    self.names = [names[i] for i in self.idx]
    self.res_name = [labels[i] for i in self.idx]
    self.res = np.array([res[i] for i in self.idx])
    self.chain = np.array([chain[i] for i in self.idx])
    self.element = [_symbol(topo['element'][i]) for i in self.idx]
    self.bonds = sorted({_key(where[a], where[b]) for a, b in bonds if a in where and b in where})
    # residues in topology order: (first, last + 1), contiguous
    self.residues, lo = [], 0
    for k in range(1, len(self.idx) + 1):
      if k == len(self.idx) or self.res[k] != self.res[lo] or self.chain[k] != self.chain[lo]:
        self.residues.append((lo, k))
        lo = k
    self.ref_atoms = {r: frozenset(n for n, s in zip(ccd.get(r)['_chem_comp_atom.atom_id'],
                                                     ccd.get(r)['_chem_comp_atom.type_symbol'])
                                   if s != 'H' and n != 'OXT') for r in SIDECHAIN_BONDS}

  def ligand_residues(self):
    lig_chains = {c for c in set(self.chain.tolist())
                  if not any(self.res_name[lo] in SIDECHAIN_BONDS for lo, hi in self.residues
                             if self.chain[lo] == c)}
    return {k for k, (lo, hi) in enumerate(self.residues)
            if self.res_name[lo] in LIGAND_RESNAMES or self.chain[lo] in lig_chains}

  def alpha_backbone_residues(self):
    res_of = np.zeros(len(self.idx), np.int64)
    for k, (lo, hi) in enumerate(self.residues):
      res_of[lo:hi] = k
    bonded = {}
    for a, b in self.bonds:
      if res_of[a] == res_of[b]:
        bonded.setdefault(res_of[a], set()).add(_key(self.names[a], self.names[b]))
    need = {('CA', 'N'), ('C', 'CA'), ('C', 'O')}
    return {k for k, (lo, hi) in enumerate(self.residues)
            if self.res_name[lo] in SIDECHAIN_BONDS or need <= bonded.get(k, set())}


def _window(p, i, j):
  e = sorted((p.element[i], p.element[j]))
  if e == ['S', 'S']:
    return DISULFIDE_BOND_WINDOW
  return SIDECHAIN_BOND_WINDOWS.get((e[0], e[1]), GENERIC_BOND_WINDOW)


def _check_topology(p, extra):
  lig, bb = p.ligand_residues(), p.alpha_backbone_residues()
  dup = mismatch = 0
  expected = set()
  tag = lambda k, lo, n: '%s%d-%s' % (p.res_name[lo], k, n)
  for c in sorted(set(p.chain.tolist())):
    prev_c = None
    for k, (lo, hi) in enumerate(p.residues):
      if p.chain[lo] != c:
        continue
      if k in lig:
        prev_c = None
        continue
      names = p.names[lo:hi]
      dup += len(names) != len(set(names))
      std = p.res_name[lo] in SIDECHAIN_BONDS
      if std:
        mismatch += set(names) != set(p.ref_atoms[p.res_name[lo]])
      if k in bb:
        bonds = {('N', 'CA'), ('CA', 'C'), ('C', 'O')} | (SIDECHAIN_BONDS[p.res_name[lo]] if std else set())
        expected.update(_key(tag(k, lo, a), tag(k, lo, b)) for a, b in bonds)
        if prev_c is not None:
          expected.add(_key(prev_c, tag(k, lo, 'N')))
        prev_c = tag(k, lo, 'C')
      else:
        prev_c = None
  res_of = np.zeros(len(p.idx), np.int64)
  for k, (lo, hi) in enumerate(p.residues):
    res_of[lo:hi] = k
  label = lambda a: '%s%d-%s' % (p.res_name[a], res_of[a], p.names[a])
  for a, b in extra:
    if p.res_name[a] in SIDECHAIN_BONDS and p.res_name[b] in SIDECHAIN_BONDS:
      expected.add(_key(label(a), label(b)))
  current = set()
  for a, b in p.bonds:
    names = _key(label(a), label(b))
    if ((p.res_name[a] in SIDECHAIN_BONDS and p.res_name[b] in SIDECHAIN_BONDS
         and res_of[a] not in lig and res_of[b] not in lig) or names in expected):
      current.add(names)
  return {'duplicate_atoms': dup, 'atoms_mismatch': mismatch,
          'bonds_mismatch': len(current ^ expected)}


def _geometry(p):
  lig, bb = p.ligand_residues(), p.alpha_backbone_residues()
  main_pairs, main_win, static, ridx_of = [], [], [], {}
  covered, side, degen, degen_res = set(), [], [], []
  for c in sorted(set(p.chain.tolist())):
    prev_c = None
    for k, (lo, hi) in enumerate(p.residues):
      if p.chain[lo] != c:
        continue
      if k in lig:
        prev_c = None
        continue
      ri = len(static)
      ridx_of[k] = ri
      by = {p.names[a]: a for a in range(lo, hi)}
      std = p.res_name[lo] in SIDECHAIN_BONDS
      expected = (p.ref_atoms[p.res_name[lo]] - set(PROT_BACKBONE)) if std else None
      static.append(expected is not None and (bool(expected - set(by)) or len(by) != hi - lo))
      if k in bb:
        for (a, b), w in MAINCHAIN_BOND_WINDOWS.items():
          if a in by and b in by:
            main_pairs.append((by[a], by[b]))
            main_win.append(w)
        if prev_c is not None and 'N' in by:
          main_pairs.append((prev_c, by['N']))
          main_win.append(PEPTIDE_BOND_WINDOW)
        prev_c = by.get('C')
      else:
        prev_c = None
      if expected is None:
        continue
      for a, b in SIDECHAIN_BONDS[p.res_name[lo]]:
        if a in by and b in by:
          side.append((ri, by[a], by[b], _window(p, by[a], by[b])))
          covered.add(_key(by[a], by[b]))
      atoms = list(by.values())
      for x, ai in enumerate(atoms[:-1]):
        for aj in atoms[x + 1:]:
          degen.append((ai, aj))
          degen_res.append(ri)
  covered.update(_key(a, b) for a, b in main_pairs)
  res_of = np.zeros(len(p.idx), np.int64)
  for k, (lo, hi) in enumerate(p.residues):
    res_of[lo:hi] = k
  for a, b in p.bonds:
    ri, rj = ridx_of.get(res_of[a]), ridx_of.get(res_of[b])
    if (a, b) in covered or ri is None or rj is None:
      continue
    covered.add((a, b))
    w = _window(p, a, b)
    for r in {ri, rj}:
      side.append((r, a, b, w))
  side.sort(key=lambda e: e[0])
  arr = lambda x, d: np.array(x, d).reshape(-1, 2)
  return dict(main_pairs=arr(main_pairs, np.int64), main_win=arr(main_win, float),
              static=np.array(static, bool), side_pairs=arr([e[1:3] for e in side], np.int64),
              side_win=arr([e[3] for e in side], float), side_res=np.array([e[0] for e in side], np.int64),
              degen=arr(degen, np.int64), degen_res=np.array(degen_res, np.int64))


def _dsq(c, pairs):
  d = c[..., pairs[:, 0], :] - c[..., pairs[:, 1], :]
  return np.einsum('...j,...j->...', d, d)


def _outside(c, pairs, win):
  d = np.sqrt(_dsq(c, pairs))
  return (d < win[:, 0]) | (d > win[:, 1])


def _flag(bad, flags, residues):
  if not len(residues):
    return
  starts = np.flatnonzero(np.r_[True, residues[1:] != residues[:-1]])
  bad[:, residues[starts]] |= np.logical_or.reduceat(flags, starts, axis=-1)


def _bonded_index(pairs, bonded):
  out = np.full(len(pairs), -1, np.int64)
  if not len(bonded) or not len(pairs):
    return out
  n = int(max(pairs.max(), bonded.max())) + 1
  kb = np.minimum(bonded[:, 0], bonded[:, 1]) * n + np.maximum(bonded[:, 0], bonded[:, 1])
  order = np.argsort(kb)
  sk = kb[order]
  kp = np.minimum(pairs[:, 0], pairs[:, 1]) * n + np.maximum(pairs[:, 0], pairs[:, 1])
  pos = np.minimum(np.searchsorted(sk, kp), len(sk) - 1)
  found = sk[pos] == kp
  out[found] = order[pos[found]]
  return out


def _candidate_pairs(c, r):
  from scipy.spatial import cKDTree
  n = c.shape[1]
  pairs = np.concatenate([cKDTree(f).query_pairs(r, output_type='ndarray').reshape(-1, 2)
                          for f in c]).astype(np.int64)
  keys = np.unique(pairs[:, 0] * n + pairs[:, 1])
  return np.stack(np.divmod(keys, n), -1)


def _count_atoms(mask, pairs, n):
  f, k = np.nonzero(mask)
  off = f * n
  flat = np.concatenate([off + pairs[k, 0], off + pairs[k, 1]])
  return np.bincount(flat, minlength=mask.shape[0] * n).reshape(mask.shape[0], n)


def _geometry_stats(p, coords):
  lig = p.ligand_residues()
  res_of = np.zeros(len(p.idx), np.int64)
  for k, (lo, hi) in enumerate(p.residues):
    res_of[lo:hi] = k
  is_lig = np.isin(res_of, list(lig))
  prot = ~is_lig
  prot_idx = np.flatnonzero(prot)
  g = _geometry(p)
  bonded = np.array(p.bonds, np.int64).reshape(-1, 2)
  sulfur = bonded[np.array([p.element[a] == p.element[b] == 'S' for a, b in p.bonds], bool)] \
      if len(bonded) else bonded
  bond_nb = np.array([max(CUTOFFS['isolated_atom'], _window(p, a, b)[1]) for a, b in p.bonds], float)
  cys_s = np.array([p.res_name[a] == 'CYS' and p.names[a] == 'SG' and p.element[a] == 'S'
                    for a in range(len(p.idx))], bool)
  elig = np.array([not is_lig[a] and p.names[a] not in BACKBONE_ATOMS and (
      p.res_name[a] not in p.ref_atoms or p.names[a] in (p.ref_atoms[p.res_name[a]] - set(PROT_BACKBONE)))
      for a in range(len(p.idx))], bool)
  max_nb = np.array([MAX_HEAVY_NEIGHBORS.get(e, 4) for e in p.element], np.int64)
  max_d = max(CUTOFFS['too_close'], CUTOFFS['inferred_bond'], CUTOFFS['sidechain_clash'],
              CUTOFFS['isolated_atom'], bond_nb.max(initial=0.0), DISULFIDE_BOND_WINDOW[1])
  T = coords.shape[0]
  st = {k: np.zeros(T, np.int64) for k in (
      'bad_bonds_backbone', 'bad_bonds_sidechain', 'bad_bonds_sulfur', 'within_residue_clashes',
      'valence_exceeded', 'isolated_atoms', 'same_chain_residue_clashes', 'cross_chain_residue_clashes',
      'non_finite_atoms')}
  for s in range(0, T, 128):
    c = np.asarray(coords[s:s + 128], float)
    fin_atom = np.isfinite(c).all(-1)
    fin = fin_atom.all(-1)
    frames = s + np.arange(len(c))
    st['non_finite_atoms'][frames] = np.count_nonzero(~fin_atom, -1)
    chunk, c = frames[fin], c[fin]
    if not len(chunk):
      continue
    st['bad_bonds_backbone'][chunk] = np.count_nonzero(_outside(c, g['main_pairs'], g['main_win']), -1)
    bad = np.repeat(g['static'][None], len(c), 0)
    if len(g['side_pairs']):
      _flag(bad, _outside(c, g['side_pairs'], g['side_win']), g['side_res'])
    if len(g['degen']):
      _flag(bad, _dsq(c, g['degen']) < CUTOFFS['degenerate'] ** 2, g['degen_res'])
    st['bad_bonds_sidechain'][chunk] = np.count_nonzero(bad, -1)
    st['bad_bonds_sulfur'][chunk] = np.count_nonzero(
        _outside(c, sulfur, np.array([DISULFIDE_BOND_WINDOW])), -1) if len(sulfur) else 0
    pairs = _candidate_pairs(c, max_d)
    d2 = _dsq(c, pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    pp = prot[i] & prot[j]
    same = p.chain[i] == p.chain[j]
    bi = _bonded_index(pairs, bonded)
    clash_pair = (res_of[i] != res_of[j]) & (bi < 0)
    nb_d = np.append(bond_nb, CUTOFFS['isolated_atom'])[bi]
    clash = d2 < CUTOFFS['sidechain_clash'] ** 2
    ss = cys_s[i] & cys_s[j] & (res_of[i] != res_of[j])
    for f in range(len(c)):
      poss = np.flatnonzero(ss & (d2[f] >= DISULFIDE_BOND_WINDOW[0] ** 2) & (d2[f] <= DISULFIDE_BOND_WINDOW[1] ** 2))
      if len(poss):
        ends = pairs[poss]
        partners = np.bincount(ends.ravel(), minlength=len(p.idx))
        clash[f, poss[np.all(partners[ends] == 1, 1)]] = False
    st['within_residue_clashes'][chunk] = np.count_nonzero(pp & (d2 < CUTOFFS['too_close'] ** 2), 1)
    inferred = _count_atoms(pp & (d2 <= CUTOFFS['inferred_bond'] ** 2), pairs, len(p.idx))
    st['valence_exceeded'][chunk] = np.count_nonzero(inferred > max_nb, 1)
    nbc = _count_atoms(pp & (d2 <= nb_d ** 2), pairs, len(p.idx))[:, prot_idx]
    st['isolated_atoms'][chunk] = np.count_nonzero(nbc == 0, 1)
    st['same_chain_residue_clashes'][chunk] = np.count_nonzero(
        same & clash_pair & elig[i] & elig[j] & clash, 1)
    st['cross_chain_residue_clashes'][chunk] = np.count_nonzero(~same & clash_pair & clash, 1)
  return st


def check(topo, bonds, ccd, coords, mode='high_quality', declared=()):
  """Per sample: (passed, first failed check or None, stats). `coords`
  (samples, atoms, 3) angstrom in BioEmu2's atom order; `bonds` the vendor
  topology's (steering.Graph.bonds); `declared` the system's declared bonds
  (atom index pairs), which the topology check expects."""
  T = coords.shape[0]
  if mode == 'no_filter':
    return np.ones(T, bool), [None] * T, {}
  p = _Protein(topo, bonds, ccd)
  if not len(p.idx):
    return np.ones(T, bool), [None] * T, {}
  where = {int(o): k for k, o in enumerate(p.idx)}
  extra = {(where[a], where[b]) for a, b in declared if a in where and b in where}
  stats = _geometry_stats(p, coords[:, p.idx])
  stats.update({k: np.full(T, v, np.int64) for k, v in _check_topology(p, extra).items()})
  th = dict(THRESHOLDS[mode])
  per_res = th.pop('per_residue')
  n_res = len(p.residues)
  ok = np.ones(T, bool)
  reasons = [None] * T
  nf = stats['non_finite_atoms'] > 0
  for f in np.flatnonzero(nf):
    reasons[f] = 'non_finite_atoms'
  ok &= ~nf
  for k, v in th.items():
    val = stats[k] / n_res if per_res else stats[k]
    bad = val > v
    for f in np.flatnonzero(bad & ok):
      reasons[f] = k
    ok &= ~bad
  return ok, reasons, stats


def annotate(inference_results, fold_input, ccd=None, mode=None):
  """BioEmu2's physicality report on each predicted structure, into its metadata
  as `physicality` = {status, reason, mode, stats}: status `passed`,
  `rejected` or `not_checked`, as BioEmu2's physicality.json. BIOEMU2_PHYSICALITY
  picks the preset (high_quality, the vendor's default; no_violations;
  high_quality_per_residue; no_filter to skip). Nothing is removed."""
  import os
  from alphafold3.model.pipeline import bioemu2_features as bf
  from alphafold3.model.pipeline import bioemu2_steering as st
  from alphafold3.constants import chemical_components
  # the CCD the featuriser read (model_features._bioemu2_features), so the atom
  # names here are the ones the samples were written with
  ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
  mode = mode or os.environ.get('BIOEMU2_PHYSICALITY', 'high_quality')
  if mode not in (*THRESHOLDS, 'no_filter'):
    raise ValueError(f'BIOEMU2_PHYSICALITY={mode!r}: want {sorted(THRESHOLDS)} or no_filter')
  _, keys, topo = bf.featurise(fold_input, ccd, mode=bf.mode(), smiles_templates=bf.SMILES_TEMPLATES,
                               return_topology=True)
  bonds = st.Graph(topo, ccd).bonds
  where = {k: i for i, k in enumerate(keys)}
  declared = [(where[a], where[b]) for a, b in (fold_input.bonded_atom_pairs or [])]
  frames = []
  for r in inference_results:
    s = r.predicted_structure
    xyz = {(str(c), int(i), str(a)): (x, y, z) for c, i, a, x, y, z in zip(
        s.chain_id, s.res_id, s.atom_name, s.atom_x, s.atom_y, s.atom_z)}
    frames.append([xyz.get(k, (np.nan,) * 3) for k in keys])
  coords = np.asarray(frames, np.float32).astype(np.float64)
  ok, reasons, stats = check(topo, bonds, ccd, coords, mode, declared)
  checked = mode != 'no_filter' and bool(stats)
  for f, r in enumerate(inference_results):
    r.metadata['physicality'] = {
        'status': ('passed' if ok[f] else 'rejected') if checked else 'not_checked',
        'reason': reasons[f], 'mode': mode,
        'stats': {k: int(v[f]) for k, v in stats.items()}}
