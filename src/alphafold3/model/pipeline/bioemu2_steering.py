"""BioEmu2's steering potentials: the topology tables, built on the host.

A transcription of bioemu2 @7846769's denoising/steering/{feats,
geometry_topology, rdkit_constraints}.py and the `prepare()` half of
potentials.py, in numpy, from the topology pipeline/bioemu2_features.py
returns. network/bioemu2_steering.py evaluates the potentials on these tables.

Two of its inputs are other people's data, fetched and cached on first use
rather than redistributed here:

  * the bond-length/-angle reference tables (`stereo_chemistry.yaml`, CCP4
    MON_LIB via OpenStructure, LGPL-3.0, with geostd's link data), from
    BioEmu2's repository at the commit this port follows;
  * Boltz-1's CCD pickle (RDKit molecules, boltz-community/boltz-1), which the
    ligand (PoseBusters, chirality, stereo-bond, planarity) tables read.

The bond graph is the vendor's `topology_from_structure`: MDTraj's standard
residue bonds -- which for every residue BioEmu2 builds are the CCD's own
heavy-atom bonds (`_standard_bonds` below; checked against the vendor's graph
by the parity harness) -- plus O3'-P links between adjacent nucleotides, C-N
links between adjacent protein residues, and the explicit template and
declared bonds.
"""

import collections
import itertools
import math
import os

import numpy as np

COMMIT = '7846769'
STEREO_URL = ('https://raw.githubusercontent.com/microsoft/bioemu2/%s/src/bioemu2/assets/'
              'stereo_chemistry.yaml' % COMMIT)
BOLTZ_CCD_URL = 'https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl'
NONBONDED_EXCLUSION_GRAPH_DISTANCE = 3
PROTEIN, DNA, RNA, NONPOLYMER = 0, 1, 2, 3
UNK_PROT = 22
CANONICAL_PROTEIN = tuple(range(2, 22))          # ALA .. VAL tokens
NUCLEIC_TO_CCD = {28: 'DA', 30: 'DC', 29: 'DG', 31: 'DT', 23: 'A', 25: 'C', 24: 'G', 26: 'U'}
TOKEN_NAMES = {2: 'ALA', 3: 'ARG', 4: 'ASN', 5: 'ASP', 6: 'CYS', 7: 'GLN', 8: 'GLU', 9: 'GLY',
               10: 'HIS', 11: 'ILE', 12: 'LEU', 13: 'LYS', 14: 'MET', 15: 'PHE', 16: 'PRO',
               17: 'SER', 18: 'THR', 19: 'TRP', 20: 'TYR', 21: 'VAL', 22: 'UNK_PROT'}
D_AMINO_ACID_TO_L = {
    'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN',
    'DHI': 'HIS', 'DIL': 'ILE', 'DLE': 'LEU', 'DLY': 'LYS', 'DPN': 'PHE', 'DPR': 'PRO',
    'DSG': 'ASN', 'DSN': 'SER', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL',
    'MED': 'MET'}
NUCLEIC_ACID_LINK_KEY = 'NA'
BOND_FALLBACK = (1.1, 2.2, 0.03)
ANGLE_FALLBACK = (60.0, 180.0, 3.0)
DISULFIDE_ANGLE = (104.2, 2.1)
_PEPTIDE_CONTEXTS = ('PEPTIDE', 'GLY', 'PRO_TRANS', 'PRO_CIS')
# MDTraj ships standard bonds for these residue names (and none for UNK, a
# modified residue or a ligand)
_STANDARD = set(TOKEN_NAMES.values()) - {'UNK_PROT'} | {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}

VDW_RADII = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
    2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
    2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
    2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
    2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
    2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
    2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
    2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]  # fmt: skip


# --- data fetched on first use -----------------------------------------------------------

def _cache_dir():
  d = os.path.expanduser(os.environ.get('BIOEMU2_STEERING_CACHE', '~/.cache/alphafold3/bioemu2'))
  os.makedirs(d, exist_ok=True)
  return d


def _fetch(url, name):
  path = os.path.join(_cache_dir(), name)
  if not os.path.exists(path):
    import urllib.request
    tmp = path + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
  return path


_STEREO = None


def stereo_tables():
  """load_stereo_tables: {bonds, angles, link_bonds, link_angles} ->
  {residue: {canonical atom-name tuple: (mean, sigma)}}."""
  global _STEREO
  if _STEREO is None:
    import yaml
    with open(_fetch(STEREO_URL, 'stereo_chemistry.%s.yaml' % COMMIT)) as fh:
      data = yaml.safe_load(fh)
    raw = {'bonds': data['monlib']['bond_data'], 'angles': data['monlib']['angle_data'],
           'link_bonds': data['monlib_link']['bond_data'],
           'link_angles': data['monlib_link']['angle_data']}
    _STEREO = {name: {res: {_canonical(tuple(k.split('_'))): (float(m), float(s))
                            for k, (m, s) in entries.items()}
                      for res, entries in table.items()}
               for name, table in raw.items()}
  return _STEREO


_BOLTZ_CCD = None


def boltz_ccd():
  global _BOLTZ_CCD
  if _BOLTZ_CCD is None:
    import pickle
    with open(_fetch(BOLTZ_CCD_URL, 'boltz1_ccd.pkl'), 'rb') as fh:
      _BOLTZ_CCD = pickle.load(fh)
  return _BOLTZ_CCD


def _canonical(atoms):
  return min(atoms, atoms[::-1])


# --- the bonded graph (geometry_topology.build_bonded_geometry_graph) ----------------------

def _standard_bonds(ccd, label, names):
  """MDTraj's intra-residue standard bonds for a residue named `label`: the
  CCD's heavy-atom bonds between the atoms present."""
  if label not in _STANDARD:
    return []
  d = ccd.get(label)
  where = {n: i for i, n in enumerate(names)}
  out = []
  for a, b in zip(d.get('_chem_comp_bond.atom_id_1') or [], d.get('_chem_comp_bond.atom_id_2') or []):
    if a in where and b in where:
      out.append((where[a], where[b]))
  return out


class Graph:
  """BondedGeometryGraph."""

  def __init__(self, topo, ccd):
    names, labels = topo['names'], topo['labels']
    res, mol, chain = topo['residue_idx'], topo['mol_type'], topo['atom_asym']
    rtype, tok = topo['residue_type'], topo['token_idx_at_atom']
    n = len(names)
    bonds = set()
    residues = []                      # (first, last + 1) per residue, in order
    lo = 0
    for i in range(1, n + 1):
      if i == n or res[i] != res[lo]:
        residues.append((lo, i))
        lo = i
    for lo, hi in residues:
      for a, b in _standard_bonds(ccd, labels[lo], names[lo:hi]):
        bonds.add((min(lo + a, lo + b), max(lo + a, lo + b)))
    for a, b in np.asarray(topo['bond_index']).reshape(2, -1).T:     # template and declared bonds
      bonds.add((int(min(a, b)), int(max(a, b))))
    # O3'-P between adjacent nucleotides of a nucleic-acid chain; C-N between
    # adjacent protein residues (whatever their template)
    nucleic_chains = {chain[lo] for lo, _ in residues if mol[lo] in (DNA, RNA)}
    prev = None
    for lo, hi in residues:
      if prev is not None and chain[prev[0]] == chain[lo]:
        pa = dict((names[i], i) for i in range(*prev))
        ca = dict((names[i], i) for i in range(lo, hi))
        if chain[lo] in nucleic_chains and "O3'" in pa and 'P' in ca:
          bonds.add((min(pa["O3'"], ca['P']), max(pa["O3'"], ca['P'])))
        if mol[prev[0]] == PROTEIN and mol[lo] == PROTEIN and 'N' in pa and 'C' in pa and 'N' in ca:
          bonds.add((min(pa['C'], ca['N']), max(pa['C'], ca['N'])))
      prev = (lo, hi)
    self.bonds = sorted(bonds)
    # atom labels: (reference residue name, atom name)
    self.labels, self.residue, self.mol = {}, {}, dict(enumerate(mol.tolist()))
    self.ordinary_x = set()
    for lo, hi in residues:
      token = int(rtype[lo]) if mol[lo] == PROTEIN else None
      is_x = token == UNK_PROT and len({int(tok[i]) for i in range(lo, hi)}) == 1
      name = NUCLEIC_TO_CCD.get(int(rtype[lo]), D_AMINO_ACID_TO_L.get(labels[lo], labels[lo]))
      for i in range(lo, hi):
        self.labels[i] = (TOKEN_NAMES[token] if is_x else name, names[i])
        self.residue[i] = (int(chain[lo]), int(res[lo]))
        if is_x:
          self.ordinary_x.add(i)
    nb = collections.defaultdict(set)
    for i, j in self.bonds:
      nb[i].add(j)
      nb[j].add(i)
    self.neighbors = nb
    self.angles = [(i, k, c) for c, bonded in sorted(nb.items())
                   for i, k in itertools.combinations(sorted(bonded), 2)]
    self.peptide_edges = {
        (i, j) for i, j in self.bonds
        if self.residue[i] != self.residue[j] and self.residue[i][0] == self.residue[j][0]
        and self.mol[i] == self.mol[j] == PROTEIN
        and {self.labels[i][1], self.labels[j][1]} == {'C', 'N'}
        and abs(self.residue[i][1] - self.residue[j][1]) == 1}

  def peptide_following_residue(self, i, j):
    if (min(i, j), max(i, j)) not in self.peptide_edges:
      return None
    res_i, name_i = self.labels[i]
    return res_i if name_i == 'N' else self.labels[j][0]

  def peptide_angle_following_residue(self, oi, c, ok):
    following = None
    if self.residue[c] == self.residue[ok]:
      following = self.peptide_following_residue(oi, c)
    if following is None and self.residue[c] == self.residue[oi]:
      following = self.peptide_following_residue(c, ok)
    return following

  def is_na_link_bond(self, i, j):
    return (self.residue[i] != self.residue[j] and self.mol[i] == self.mol[j]
            and self.mol[i] in (DNA, RNA)
            and {self.labels[i][1], self.labels[j][1]} == {"O3'", 'P'}
            and self.residue[i][0] == self.residue[j][0]
            and abs(self.residue[i][1] - self.residue[j][1]) == 1)

  def is_disulfide(self, i, j):
    return self.labels[i] == self.labels[j] == ('CYS', 'SG') and self.mol[i] == self.mol[j] == PROTEIN

  def is_na_link_angle(self, oi, c, ok):
    atoms = (oi, c, ok)
    return len({self.mol[a] for a in atoms}) == 1 and any(
        self.residue[c] == self.residue[other] and self.is_na_link_bond(outer, c)
        for outer, other in ((oi, ok), (ok, oi)))

  def nonbonded_exclusions(self, max_bonds=NONBONDED_EXCLUSION_GRAPH_DISTANCE):
    out = set()
    for source in self.neighbors:
      depth = {source: 0}
      queue = collections.deque([source])
      while queue:
        node = queue.popleft()
        if depth[node] < max_bonds:
          for other in self.neighbors[node]:
            if other not in depth:
              depth[other] = depth[node] + 1
              queue.append(other)
      out.update((source, o) for o in depth if source < o)
    return out


def polymer_steric_eligible(graph, topo):
  """build_polymer_steric_graph's endpoint mask."""
  rtype, mol = topo['residue_type'], topo['mol_type']
  x = np.zeros(len(rtype), bool)
  x[sorted(graph.ordinary_x)] = True
  protein = mol == PROTEIN
  canon_p = np.isin(rtype, CANONICAL_PROTEIN)
  nucleic = np.isin(mol, (DNA, RNA))
  canon_n = np.isin(rtype, list(NUCLEIC_TO_CCD))
  return (protein & (canon_p | x)) | (nucleic & canon_n)


# --- bond/angle references (geometry_topology.get_*_reference) ----------------------------

def _lookup(table, residue, atoms):
  d = table.get(residue)
  return None if d is None else d.get(_canonical(atoms))


def _peptide_link_key(following):
  return 'GLY' if following == 'GLY' else 'PRO_TRANS' if following == 'PRO' else 'PEPTIDE'


def _pooled(table, names, peptide):
  ids = _PEPTIDE_CONTEXTS if peptide else tuple(TOKEN_NAMES[t] for t in CANONICAL_PROTEIN)
  entries = [r for i in ids if (r := _lookup(table, i, names)) is not None]
  if not entries:
    return None
  return min(m for m, _ in entries), max(m for m, _ in entries), max(s for _, s in entries)


def bond_reference(g, i, j, tables, pooled):
  names = (g.labels[i][1], g.labels[j][1])
  same = g.residue[i] == g.residue[j]
  following = g.peptide_following_residue(i, j)
  if i in g.ordinary_x or j in g.ordinary_x:
    if same or following is not None:
      key = (_canonical(names), following is not None)
      if key not in pooled:
        pooled[key] = _pooled(tables['link_bonds'] if following is not None else tables['bonds'],
                              names, following is not None)
      return pooled[key] or BOND_FALLBACK
    return BOND_FALLBACK
  if following is not None:
    ref = _lookup(tables['link_bonds'], _peptide_link_key(following), names)
  elif g.is_na_link_bond(i, j):
    ref = _lookup(tables['link_bonds'], NUCLEIC_ACID_LINK_KEY, names)
  elif same:
    ref = _lookup(tables['bonds'], g.labels[i][0], names)
  elif g.is_disulfide(i, j):
    ref = _lookup(tables['link_bonds'], 'PEPTIDE', names)
  else:
    ref = None
  return (ref[0], ref[0], ref[1]) if ref is not None else BOND_FALLBACK


def angle_reference(g, oi, c, ok, tables, pooled):
  atoms = (oi, c, ok)
  names = tuple(g.labels[a][1] for a in atoms)
  same = len({g.residue[a] for a in atoms}) == 1
  following = g.peptide_angle_following_residue(oi, c, ok)
  peptide = following is not None
  if any(a in g.ordinary_x for a in atoms):
    if same or peptide:
      key = (_canonical(names), peptide)
      if key not in pooled:
        pooled[key] = _pooled(tables['link_angles'] if peptide else tables['angles'], names, peptide)
      return pooled[key] or ANGLE_FALLBACK
    return ANGLE_FALLBACK
  if following is not None:
    ref = _lookup(tables['link_angles'], _peptide_link_key(following), names)
  elif g.is_na_link_angle(oi, c, ok):
    ref = _lookup(tables['link_angles'], NUCLEIC_ACID_LINK_KEY, names)
  elif same:
    ref = _lookup(tables['angles'], g.labels[c][0], names)
  elif _canonical(names) == ('CB', 'SG', 'SG') and any(g.is_disulfide(c, o) for o in (oi, ok)):
    ref = DISULFIDE_ANGLE
  else:
    ref = None
  return (ref[0], ref[0], ref[1]) if ref is not None else ANGLE_FALLBACK


def geometry_bounds(reference, tolerance, angle=False):
  """compute_geometry_bounds, in float32 as the vendor's torch arithmetic."""
  lo, hi, sigma = (tuple(math.radians(v) for v in reference) if angle else reference)
  width = np.float32(sigma) * np.float32(tolerance)
  lower = float(np.float32(lo) - width)
  upper = float(np.float32(hi) + width)
  return max(0.0, lower), min(math.pi if angle else math.inf, upper)


# --- RDKit ligand constraints (rdkit_constraints.py) -------------------------------------

def _rdkit_constraints(mol, idx_map):
  from rdkit import Chem
  from rdkit.Chem import HybridizationType
  from rdkit.Chem.rdchem import BondStereo
  from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
  geo, chir, ster, plan = [], [], [], []
  if mol.GetNumAtoms() > 1:
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    b = GetMoleculeBoundsMatrix(mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True,
                                useMacrocycle14config=False)
    bonds = set(tuple(sorted(x)) for x in mol.GetSubstructMatches(Chem.MolFromSmarts('*~*')))
    angles = set(tuple(sorted([a[0], a[2]])) for a in mol.GetSubstructMatches(Chem.MolFromSmarts('*~*~*')))
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
      if i in idx_map and j in idx_map:
        geo.append(((idx_map[i], idx_map[j]), tuple(sorted([i, j])) in bonds,
                    tuple(sorted([i, j])) in angles, b[i, j], b[j, i]))
  if all(a.HasProp('_CIPRank') for a in mol.GetAtoms()):
    for c_idx, orient in Chem.FindMolChiralCenters(mol, includeUnassigned=False):
      center = mol.GetAtomWithIdx(c_idx)
      nb = sorted(((x.GetIdx(), int(x.GetProp('_CIPRank'))) for x in center.GetNeighbors()),
                  key=lambda x: x[1], reverse=True)
      nb = tuple(x[0] for x in nb)
      is_r = orient == 'R'
      if len(nb) > 4 or center.GetHybridization() != HybridizationType.SP3:
        continue
      ids = (*nb[:3], c_idx)
      if all(i in idx_map for i in ids):
        chir.append((tuple(idx_map[i] for i in ids), is_r))
      if len(nb) == 4:
        for skip in range(3):
          cs = nb[:skip] + nb[skip + 1:]
          ids = (cs[::-1] if skip % 2 == 0 else cs) + (c_idx,)
          if all(i in idx_map for i in ids):
            chir.append((tuple(idx_map[i] for i in ids), is_r))
    for bond in mol.GetBonds():
      st = bond.GetStereo()
      if st in {BondStereo.STEREOE, BondStereo.STEREOZ}:
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sn = [x[0] for x in sorted(((x.GetIdx(), int(x.GetProp('_CIPRank')))
                                    for x in mol.GetAtomWithIdx(s).GetNeighbors() if x.GetIdx() != e),
                                   key=lambda x: x[1], reverse=True)]
        en = [x[0] for x in sorted(((x.GetIdx(), int(x.GetProp('_CIPRank')))
                                    for x in mol.GetAtomWithIdx(e).GetNeighbors() if x.GetIdx() != s),
                                   key=lambda x: x[1], reverse=True)]
        is_e = st == BondStereo.STEREOE
        ids = (sn[0], s, e, en[0])
        if all(i in idx_map for i in ids):
          ster.append((tuple(idx_map[i] for i in ids), is_e))
        if len(sn) == 2 and len(en) == 2:
          ids = (sn[1], s, e, en[1])
          if all(i in idx_map for i in ids):
            ster.append((tuple(idx_map[i] for i in ids), is_e))
  smarts = Chem.MolFromSmarts('[C;X3;^2](*)(*)=[C;X3;^2](*)(*)')
  for m in mol.GetSubstructMatches(smarts):
    if all(i in idx_map for i in m):
      plan.append(tuple(idx_map[i] for i in m))
  return geo, chir, ster, plan


def _ligand_tables(topo, ccd_mols):
  """Tier 2: per non-polymer residue of a CCD-coded Ligand chain."""
  names, res, mol, chain = topo['names'], topo['residue_idx'], topo['mol_type'], topo['atom_asym']
  n = len(names)
  geo, chir, ster, plan = [], [], [], []
  lo = 0
  while lo < n:
    hi = lo + 1
    while hi < n and res[hi] == res[lo] and chain[hi] == chain[lo]:
      hi += 1
    if mol[lo] == NONPOLYMER:
      code = topo['ligand_ccd'].get(int(chain[lo]))
      ref = ccd_mols.get(code) if code else None
      if ref is not None:
        where = {names[k].strip(): k for k in range(lo, hi)}
        idx_map = {}
        for r_i, atom in enumerate(ref.GetAtoms()):
          nm = str(atom.GetPropsAsDict().get('name', atom.GetSymbol())).strip()
          if nm in where:
            idx_map[r_i] = where[nm]
        if idx_map:
          g, c, s, p = _rdkit_constraints(ref, idx_map)
          geo += g
          chir += c
          ster += s
          plan += p
    lo = hi
  t = {
      'rdkit_bounds_index': np.array([g[0] for g in geo], np.int64).reshape(-1, 2).T,
      'rdkit_bounds_bond_mask': np.array([g[1] for g in geo], bool),
      'rdkit_bounds_angle_mask': np.array([g[2] for g in geo], bool),
      'rdkit_upper_bounds': np.array([g[3] for g in geo], np.float32),
      'rdkit_lower_bounds': np.array([g[4] for g in geo], np.float32),
      'chiral_atom_index': np.array([c[0] for c in chir], np.int64).reshape(-1, 4).T,
      'chiral_atom_orientations': np.array([c[1] for c in chir], bool),
      'stereo_bond_index': np.array([s[0] for s in ster], np.int64).reshape(-1, 4).T,
      'stereo_bond_orientations': np.array([s[1] for s in ster], bool),
      'planar_bond_index': np.array(plan, np.int64).reshape(-1, 6).T,
  }
  return t


# --- everything the default potentials read ------------------------------------------------

def tables(topo, ccd, sigma_tolerance=10.0, ligand_tables=None):
  """Every potential's prepared arrays, keyed as the vendor's feats are.

  ligand_tables: Tier 2 to use instead of building it from Boltz's CCD (the
  parity harness hands over the vendor's own: RDKit releases differ in their
  bounds matrices and CIP ranks)."""
  n = len(topo['names'])
  n_pad = topo['n_pad']
  g = Graph(topo, ccd)
  chain = np.zeros(n_pad, np.int64)
  chain[:n] = topo['atom_asym']
  pad = np.zeros(n_pad, bool)
  pad[:n] = True
  radius = np.zeros(n_pad, np.float32)
  radius[:n] = np.array([0.0] + VDW_RADII, np.float32)[np.minimum(topo['element'], 118)]
  n_chains = int(topo['asym_id'].max()) + 1
  sizes = np.bincount(chain[pad], minlength=n_chains)
  out = {'atom_pad_mask': pad, 'atom_chain_id': chain, 'atom_vdw_radius': radius,
         'chain_sizes': sizes}
  # SymmetricChainCOMPotential: chain pairs of one entity
  pairs = sorted(set(zip(topo['asym_id'].tolist(), topo['entity_id'].tolist())))
  sym = [(a[0], b[0]) for x, a in enumerate(pairs) for b in pairs[x + 1:] if a[1] == b[1]]
  out['symmetric_chain_index'] = np.array(sym, np.int64).reshape(-1, 2).T
  # InterChainVDWOverlapPotential
  excl = np.array(sorted(g.nonbonded_exclusions()), np.int64).reshape(-1, 2).T
  cross = excl[:, chain[excl[0]] != chain[excl[1]]] if excl.size else excl
  out['interchain_steric_atom_mask'] = pad & (sizes > 1)[chain]
  out['interchain_bonded_pair_key'] = cross[0] * n_pad + cross[1] if cross.size else np.zeros(0, np.int64)
  # IntraChainVDWOverlapPotential / PolymerLocalClashPotential
  eligible = np.zeros(n_pad, bool)
  eligible[:n] = polymer_steric_eligible(g, topo)
  out['polymer_steric_atom_mask'] = eligible
  out['polymer_bonded_pair_key'] = excl[0] * n_pad + excl[1] if excl.size else np.zeros(0, np.int64)
  local = np.array(sorted(g.nonbonded_exclusions() - set(g.bonds)), np.int64).reshape(-1, 2).T
  if local.size:
    keep = eligible[local[0]] & eligible[local[1]] & (topo['residue_idx'][local[0]] != topo['residue_idx'][local[1]])
    local = local[:, keep]
  out['polymer_local_clash_index'] = local
  # BondLengthPotential / BondAnglePotential
  # references (lowest mean, highest mean, sigma) -- each Bond*/Angle potential
  # applies its own sigma_tolerance (network/bioemu2_steering.host_tables) --
  # and the bounds at `sigma_tolerance`, the default potentials'
  st = stereo_tables()
  pooled = {}
  out['bond_ref'] = np.array([bond_reference(g, i, j, st, pooled) for i, j in g.bonds], float).reshape(-1, 3)
  pooled = {}
  out['angle_ref'] = np.array([angle_reference(g, i, c, k, st, pooled) for i, k, c in g.angles],
                              float).reshape(-1, 3)
  bl = [geometry_bounds(r, sigma_tolerance) for r in out['bond_ref'].tolist()]
  out['bond_index'] = np.array(g.bonds, np.int64).reshape(-1, 2).T
  out['bond_lower'] = np.array([b[0] for b in bl], np.float32)
  out['bond_upper'] = np.array([b[1] for b in bl], np.float32)
  al = [geometry_bounds(r, sigma_tolerance, angle=True) for r in out['angle_ref'].tolist()]
  out['angle_index'] = np.array(g.angles, np.int64).reshape(-1, 3).T
  out['angle_lower'] = np.array([a[0] for a in al], np.float32)
  out['angle_upper'] = np.array([a[1] for a in al], np.float32)
  out['graph_bonds'] = np.array(g.bonds, np.int64).reshape(-1, 2)
  # Tier 2
  if ligand_tables is not None:
    out.update(ligand_tables)
  elif topo['ligand_ccd']:
    out.update(_ligand_tables(topo, boltz_ccd()))
  else:
    out.update(_ligand_tables({**topo, 'ligand_ccd': {}}, {}))
  return out
