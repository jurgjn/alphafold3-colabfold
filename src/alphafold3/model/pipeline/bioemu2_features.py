"""BioEmu2's input features, built from an AlphaFold 3 fold input.

network/bioemu2.py reads BioEmu2's own feature dict, not an AF3 batch, so this
module is the whole of its featurisation: a transcription of bioemu2 @7846769's
System -> AllAtomStructure -> batch path (structure/inference_inputs.py,
structure/all_atom_structure.py, msa/colabfold.py, msa/features.py,
biomoa/utils/feature_converter.py), driven by an AF3 fold input and our CCD.

Its conventions, every one different from AlphaFold 3's somewhere:

  * reference positions are the CCD's IDEAL coordinates, heavy atoms only,
    centred on the component's own mean -- deterministic, no conformer draw.
    Polymer residues are centred over the full component and then lose OXT
    (protein) or OP3 (nucleic acid) at EVERY position, the C-terminus included;
  * a standard residue is one token (centre atom CA or C1'); an unknown residue
    is ALA's template typed UNK; a modified residue, a ligand and a glycan
    monomer are one token per atom, typed UNK_PROT;
  * residue_index and ref_space_uid count residues across the whole system;
  * the MSA is Boltz-1's: rows lined up by index across chains (paired rows
    first for a heteromer that has them), 4092 rows, deletion value
    pi/2 * atan(d/3) -- not AF3's 2/pi.

How an AF3 fold input maps onto BioEmu2's System (the stock runner uses the
same rule, so both sides see one System):

  * chains with identical content form one entity, entities in order of first
    appearance, and the chains are laid out entity by entity;
  * a ligand with one CCD code and no declared bond is a Ligand; one with
    several codes or any declared bond is a Branched entity (its donor
    residues lose their leaving atoms); a SMILES ligand is a Ligand from
    SMILES, whose conformer is a random draw (see `smiles_templates`).
"""

import dataclasses
import math

import numpy as np

from alphafold3.common import folding_input

MAX_SEQS = 4092
MAX_PAIRED_SEQS = 8192
NUM_TOKENS = 33
GAP = 1
UNK_PROT = 22
WINDOW = 32
METHOD = {'equilibrium': 4, 'structure': 7}       # "md", "non-md"

_PROT = dict(zip('ARNDCEQGHILKMFPSTWYV',
                 [2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]))
_PROT.update({c: UNK_PROT for c in 'XJBZOU'})
_PROT['-'] = GAP
_RNA = {'A': 23, 'G': 24, 'C': 25, 'U': 26, 'N': 27}
_DNA = {'A': 28, 'G': 29, 'C': 30, 'T': 31, 'N': 32}
_THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
          'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
          'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
_UNKNOWN_CCD = {'protein': 'UNK', 'dna': 'DN', 'rna': 'N'}
_CAPPING = ('OXT', 'OP3')


def mode():
  """The sampling mode a run asks for: BIOEMU2_MODE, `structure` by default.
  Propagation (BIOEMU2_X0 set) is BioEmu2's equilibrium mode by definition --
  bioemu2.propagation embeds with mode="equilibrium" -- so it implies it."""
  import os
  m = os.environ.get('BIOEMU2_MODE')
  if os.environ.get('BIOEMU2_X0'):
    if m not in (None, 'equilibrium'):
      raise ValueError('propagation (BIOEMU2_X0) samples in equilibrium mode; unset BIOEMU2_MODE')
    return 'equilibrium'
  m = m or 'structure'
  if m not in METHOD:
    raise ValueError(f'BIOEMU2_MODE={m!r}: want structure or equilibrium')
  return m


def x0_from_mmcif(path, atom_keys):
  """A starting frame for propagation, in BioEmu2's atom order: every atom in
  `atom_keys` (chain, 1-based residue, name) read from the first model of an
  mmCIF -- e.g. a previous sample of this same input. A .npy is taken as the
  coordinates themselves, already in that order (how the parity harness hands
  both sides one frame)."""
  if str(path).endswith('.npy'):
    x0 = np.load(path).astype(np.float32)
    if x0.shape != (len(atom_keys), 3):
      raise ValueError(f'{path}: shape {x0.shape}, want ({len(atom_keys)}, 3)')
    return x0
  from alphafold3 import structure
  with open(path) as fh:
    s = structure.from_mmcif(fh.read())
  where = {(str(c), int(r), str(a)): (x, y, z) for c, r, a, x, y, z in zip(
      s.chain_id, s.res_id, s.atom_name, s.atom_x, s.atom_y, s.atom_z)}
  missing = [k for k in atom_keys if k not in where]
  if missing:
    raise ValueError(f'{path} lacks {len(missing)} atom(s) of this input, e.g. {missing[:3]}')
  return np.array([where[k] for k in atom_keys], np.float32)


def add_propagation(f, x0, lag_ns, steps=1):
  """Condition on a starting frame `x0` (real atoms, angstrom) at lag `lag_ns`:
  propagate_with_embeddings' coords_cond (the frame centred), dt in ps, and the
  flag. `x0` itself rides along, uncentred, for superposing the endpoint on it;
  `propagation_steps` is a length-`steps` placeholder so the chain length is a
  static shape."""
  if not (lag_ns > 0 and np.isfinite(lag_ns)):
    raise ValueError('the lag must be finite and positive')
  g = dict(f)
  n_pad = f['template_coords'].shape[0]
  frame = np.zeros((n_pad, 3), np.float32)
  frame[:len(x0)] = x0
  cond = np.zeros((n_pad, 3), np.float32)
  cond[:len(x0)] = x0 - x0.mean(0)
  g['coords_cond'] = cond
  g['x0'] = frame
  g['dt'] = np.float32(lag_ns * 1000.0)
  g['is_dynamics_conditioned'] = np.bool_(True)
  g['propagation_steps'] = np.zeros(steps, np.int32)
  return g


def encode_atom_name(name):
  return [ord(c) - 32 for c in name] + [0] * (4 - len(name))


# --- CCD templates ------------------------------------------------------------------

@dataclasses.dataclass
class Template:
  names: list
  element: np.ndarray       # atomic numbers
  charge: np.ndarray
  coords: np.ndarray        # float32, centred as the vendor centres it
  bonds: np.ndarray         # (2, n) atom indices within the template


def _atomic_number(symbol):
  if symbol.upper() == 'X':
    return 0                # mdtraj's virtual site
  from rdkit import Chem
  return Chem.GetPeriodicTable().GetAtomicNumber(symbol.capitalize())


def ccd_template(ccd, code, remove_leaving=False):
  """template_from_cifwrapper: heavy atoms (minus leaving atoms when asked),
  ideal coordinates centred on THEIR mean."""
  d = ccd.get(code)
  if d is None:
    raise ValueError(f'{code} is not in the CCD')
  names = list(d['_chem_comp_atom.atom_id'])
  sym = list(d['_chem_comp_atom.type_symbol'])
  try:
    xyz = np.array([[float(v) for v in d['_chem_comp_atom.pdbx_model_Cartn_%s_ideal' % c]]
                    for c in 'xyz'], dtype=float).T
  except ValueError:
    xyz = np.array([[float(v) for v in d['_chem_comp_atom.model_Cartn_%s' % c]]
                    for c in 'xyz'], dtype=float).T
  keep = np.array([s.upper() != 'H' for s in sym])
  if remove_leaving:
    keep &= np.array([f != 'Y' for f in d['_chem_comp_atom.pdbx_leaving_atom_flag']])
  old_to_new = np.full(len(names), -1)
  old_to_new[keep] = np.arange(keep.sum())
  pos = xyz[keep]
  pos = pos - pos.mean(0)
  charge = np.array([0.0 if q in ('?', '.') else float(q) for q in d['_chem_comp_atom.charge']])[keep]
  b1 = d.get('_chem_comp_bond.atom_id_1') or []
  b2 = d.get('_chem_comp_bond.atom_id_2') or []
  bonds = [(old_to_new[names.index(a)], old_to_new[names.index(b)]) for a, b in zip(b1, b2)]
  bonds = np.array([(a, b) for a, b in bonds if a >= 0 and b >= 0], dtype=np.int64).reshape(-1, 2).T
  return Template(names=[n for n, k in zip(names, keep) if k],
                  element=np.array([_atomic_number(s) for s, k in zip(sym, keep) if k], np.int64),
                  charge=charge.astype(np.float32), coords=pos.astype(np.float32), bonds=bonds)


def _drop(t, drop_names):
  keep = np.array([n not in drop_names for n in t.names])
  old_to_new = np.full(len(t.names), -1)
  old_to_new[keep] = np.arange(keep.sum())
  bonds = old_to_new[t.bonds] if t.bonds.size else t.bonds
  bonds = bonds[:, (bonds >= 0).all(0)] if bonds.size else bonds
  return Template(names=[n for n, k in zip(t.names, keep) if k], element=t.element[keep],
                  charge=t.charge[keep], coords=t.coords[keep], bonds=bonds)


def is_leaving_atom_donor(ccd, code, bond_atom_names):
  d = ccd.get(code)
  heavy_leaving = {a for a, f, e in zip(d['_chem_comp_atom.atom_id'],
                                         d['_chem_comp_atom.pdbx_leaving_atom_flag'],
                                         d['_chem_comp_atom.type_symbol']) if f == 'Y' and e != 'H'}
  b1 = d.get('_chem_comp_bond.atom_id_1') or []
  b2 = d.get('_chem_comp_bond.atom_id_2') or []
  names = set(bond_atom_names)
  return any((a in names and b in heavy_leaving) or (b in names and a in heavy_leaving)
             for a, b in zip(b1, b2))


def smiles_order(smiles, t):
  """For each atom of `t` (a BioEmu2 SMILES template, in its InChI-derived
  order), the index of the same atom in RDKit's SMILES order -- the order AF3
  lays a SMILES ligand out in. Matched on the heavy-atom graph."""
  from rdkit import Chem
  ref = Chem.MolFromSmiles(smiles)
  mol = Chem.RWMol()
  for z in t.element:
    mol.AddAtom(Chem.Atom(int(z)))
  for a, b in t.bonds.T:
    mol.AddBond(int(a), int(b), Chem.BondType.SINGLE)
  params = Chem.AdjustQueryParameters()
  params.makeBondsGeneric = True
  q = Chem.AdjustQueryProperties(mol.GetMol(), params)
  match = ref.GetSubstructMatch(q)
  if len(match) != len(t.element):
    raise ValueError(f'cannot match the BioEmu2 template of {smiles} onto its SMILES')
  return list(match)


def smiles_template(smiles, seed=0, max_iters=1000):
  """_template_from_rdmol on SMILES -> InChI -> mol, as the vendor does. The
  vendor's ETKDGv3 is unseeded; ours is seeded so a run is reproducible."""
  from rdkit import Chem
  from rdkit.Chem import AllChem
  from collections import Counter
  mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(smiles)))
  mol_h = Chem.AddHs(mol)
  counter = Counter()
  for idx in Chem.CanonicalRankAtoms(mol_h):
    atom = mol_h.GetAtomWithIdx(idx)
    el = atom.GetSymbol().upper()
    counter[el] += 1
    atom.SetProp('name', f'{el}{counter[el]}')
  gen = AllChem.ETKDGv3()
  gen.randomSeed = seed
  cid = AllChem.EmbedMolecule(mol_h, gen)
  if cid == -1:
    gen.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol_h, gen)
  AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=max_iters)
  mol = Chem.RemoveAllHs(mol_h)
  conf = mol.GetConformer(cid)
  n = mol.GetNumAtoms()
  bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
                   np.int64).reshape(-1, 2).T
  return Template(names=[a.GetProp('name') for a in mol.GetAtoms()],
                  element=np.array([a.GetAtomicNum() for a in mol.GetAtoms()], np.int64),
                  charge=np.array([a.GetFormalCharge() for a in mol.GetAtoms()], np.float32),
                  coords=np.array([list(conf.GetAtomPosition(i)) for i in range(n)], np.float32),
                  bonds=bonds)


# --- a chain as residues and tokens --------------------------------------------------

@dataclasses.dataclass
class _Residue:
  t: Template
  restype: int
  per_atom_tokens: bool
  centre: str | None         # the centre atom of a single-token residue
  label: str = 'UNK'         # residue_labels: what the vendor's topology names it


def _polymer_residues(ccd, kind, sequence, mods):
  """inference_input_structure_from_{protein,rna,dna}_sequence (with mods)."""
  has_mods = bool(mods)
  mods = dict((pos, code) for code, pos in mods)
  table = {'protein': _PROT, 'rna': _RNA, 'dna': _DNA}[kind]
  unk_letter = 'X' if kind == 'protein' else 'N'
  out = []
  for pos, letter in enumerate(sequence, start=1):
    code = mods.get(pos)
    if code is None and letter == unk_letter:
      code = _UNKNOWN_CCD[kind]
    if code is not None and code != _UNKNOWN_CCD[kind]:
      t = _drop(ccd_template(ccd, code), _CAPPING)          # a modified residue: atom tokens
      out.append(_Residue(t, UNK_PROT, True, None, code))
      continue
    if kind == 'protein':
      ccd_code = 'ALA' if code is not None or letter not in _THREE else _THREE[letter]
      t = _drop(ccd_template(ccd, ccd_code), ('OXT',))
      restype = table.get(letter, UNK_PROT) if code is None else UNK_PROT
      out.append(_Residue(t, restype, False, 'CA', 'UNK' if restype == UNK_PROT else ccd_code))
    else:
      ccd_code = letter if kind == 'rna' else 'D' + letter
      if code is not None:
        ccd_code = _UNKNOWN_CCD[kind]
      t = _drop(ccd_template(ccd, ccd_code), ('OP3',))
      unknown = code is not None or letter == 'N'
      # an unknown nucleotide is labelled by index_to_pdb_resname ('UNK') unless the
      # chain has modifications, when the mods path labels it with its CCD code
      label = ((ccd_code if has_mods else 'UNK') if unknown else ccd_code)
      out.append(_Residue(t, table['N'] if unknown else table[letter], False, "C1'", label))
  return out


def _entities(fold_input):
  """BioEmu2 entities from AF3 chains: [(kind, payload, [chain ids])], in order."""
  bonds = fold_input.bonded_atom_pairs or []
  bonded = {a[0] for pair in bonds for a in pair}
  ents, index = [], {}
  for ch in fold_input.chains:
    if isinstance(ch, folding_input.ProteinChain):
      key = ('protein', ch._sequence, tuple(sorted(ch.ptms)))   # pylint: disable=protected-access
    elif isinstance(ch, folding_input.RnaChain):
      key = ('rna', ch._sequence, tuple(sorted(ch.modifications)))  # pylint: disable=protected-access
    elif isinstance(ch, folding_input.DnaChain):
      key = ('dna', ch._sequence, tuple(sorted(ch.modifications)))  # pylint: disable=protected-access
    elif ch.smiles is not None:
      key = ('smiles', ch.smiles, ())
    elif len(ch.ccd_ids) == 1 and ch.id not in bonded:
      key = ('ccd', ch.ccd_ids[0], ())
    else:
      key = ('branched', tuple(ch.ccd_ids), ())
    if key not in index:
      index[key] = len(ents)
      ents.append((key, []))
    ents[index[key]][1].append(ch)
  return ents


# --- the whole system -----------------------------------------------------------------

def featurise(fold_input, ccd, mode='structure', smiles_templates=None, pad_atoms_to=None,
              pad_tokens_to=None, contacts=None, return_topology=False):
  """fold input -> (BioEmu2 feature dict without a batch axis, atom keys).

  smiles_templates: {chain id: Template} to use instead of a fresh RDKit draw
  (the parity harness passes the vendor's own draw here). atom keys are
  (chain id, 1-based residue index, atom name) for every real atom, in order --
  how an output is mapped back onto the structure. return_topology adds a third
  value: the per-atom topology steering builds its tables from (see
  pipeline/bioemu2_steering.py).
  """
  smiles_templates = smiles_templates or {}
  bonds = fold_input.bonded_atom_pairs or []
  feats = {k: [] for k in ('template_coords', 'element', 'charge', 'name_encoding',
                           'residue_idx', 'token_idx_at_atom', 'mol_type', 'residue_type')}
  tok = {k: [] for k in ('asym_id', 'entity_id', 'sym_id', 'centre')}
  bond_index, atom_keys, chain_info = [], [], []
  labels, ligand_ccd = [], {}
  n_atoms = n_tokens = n_res = 0
  chain_order = []
  for entity_id, ((kind, payload, mods), chains) in enumerate(_entities(fold_input)):
    for sym_id, ch in enumerate(chains):
      chain_order.append(ch)
      if kind in ('protein', 'rna', 'dna'):
        residues = _polymer_residues(ccd, kind, payload, mods)
        mol = {'protein': 0, 'dna': 1, 'rna': 2}[kind]
      elif kind == 'ccd':
        residues = [_Residue(ccd_template(ccd, payload), UNK_PROT, True, None, payload)]
        mol = 3
      elif kind == 'smiles':
        t = smiles_templates.get(ch.id) or smiles_template(payload)
        residues = [_Residue(t, UNK_PROT, True, None)]
        mol = 3
      else:
        atoms_by_res = {}
        for pair in bonds:
          for c, r, a in pair:
            if c == ch.id:
              atoms_by_res.setdefault(r - 1, []).append(a)
        residues = [_Residue(ccd_template(ccd, code, remove_leaving=is_leaving_atom_donor(
            ccd, code, atoms_by_res.get(i, [])) if i in atoms_by_res else False), UNK_PROT, True,
                             None, code)
                    for i, code in enumerate(payload)]
        mol = 3
      if kind == 'ccd':
        ligand_ccd[len(chain_order) - 1] = payload
      chain_tokens_start = n_tokens
      for r_local, res in enumerate(residues):
        labels.extend([res.label] * len(res.t.names))
        t = res.t
        n = len(t.names)
        feats['template_coords'].append(t.coords)
        feats['element'].append(t.element)
        feats['charge'].append(t.charge)
        feats['name_encoding'].append(np.array([encode_atom_name(x) for x in t.names], np.int64))
        feats['residue_idx'].append(np.full(n, n_res, np.int64))
        feats['mol_type'].append(np.full(n, mol, np.int64))
        feats['residue_type'].append(np.full(n, res.restype, np.int64))
        # a standard residue is built with no bonds at all; only atom-token
        # residues (ligands, modified residues, glycans) keep their CCD bonds
        if res.per_atom_tokens and t.bonds.size:
          bond_index.append(t.bonds + n_atoms)
        if res.per_atom_tokens:
          feats['token_idx_at_atom'].append(np.arange(n_tokens, n_tokens + n, dtype=np.int64))
          tok['centre'].extend(range(n_atoms, n_atoms + n))
          n_tokens += n
        else:
          feats['token_idx_at_atom'].append(np.full(n, n_tokens, np.int64))
          tok['centre'].append(n_atoms + t.names.index(res.centre))
          n_tokens += 1
        atom_keys.extend((ch.id, r_local + 1, name) for name in t.names)
        n_atoms += n
        n_res += 1
      nt = n_tokens - chain_tokens_start
      tok['asym_id'].extend([len(chain_order) - 1] * nt)
      tok['entity_id'].extend([entity_id] * nt)
      tok['sym_id'].extend([sym_id] * nt)
      chain_info.append((ch, kind, payload, residues, chain_tokens_start, n_tokens))
  f = {k: np.concatenate(v) for k, v in feats.items()}
  f['template_coords'] = f['template_coords'].astype(np.float32)
  centre = np.array(tok['centre'], np.int64)

  # declared covalent bonds (System.bonds -> _add_extra_covalent_bonds)
  key_index = {k: i for i, k in enumerate(atom_keys)}
  for (c1, r1, a1), (c2, r2, a2) in bonds:
    bond_index.append(np.array([[key_index[(c1, r1, a1)]], [key_index[(c2, r2, a2)]]], np.int64))
  bond_index = np.concatenate(bond_index, 1) if bond_index else np.zeros((2, 0), np.int64)

  T = n_tokens
  token_bonds = np.zeros((T, T), np.float32)
  tb = f['token_idx_at_atom'][bond_index]
  token_bonds[tb[0], tb[1]] = 1
  token_bonds = np.clip(token_bonds + token_bonds.T, 0, 1)

  out = {
      'res_type': f['residue_type'][centre],
      'residue_index': f['residue_idx'][centre],
      'asym_id': np.array(tok['asym_id'], np.int64),
      'entity_id': np.array(tok['entity_id'], np.int64),
      'sym_id': np.array(tok['sym_id'], np.int64),
      'token_center_atom_index': centre,
      'token_pad_mask': np.ones(T, bool),
      'token_bonds': token_bonds[..., None],
      'method_feature': np.full(T, METHOD[mode], np.int64),
      'pocket_feature': np.zeros(T, np.int64),
      'contact_conditioning': np.broadcast_to(np.eye(5, dtype=np.int64)[0], (T, T, 5)).copy(),
      'contact_threshold': np.zeros((T, T), np.float32),
      'dt': np.float32(0.0),
      'is_dynamics_conditioned': np.bool_(False),
  }
  out.update(_msa_features(chain_order, chain_info, f, centre, T))
  if contacts:
    out.update(_contact_features(contacts, chain_info, f, centre, T))

  # atoms, padded to the window (batch_collate)
  n_pad = pad_atoms_to or math.ceil(n_atoms / WINDOW) * WINDOW
  def pad(x, value=0):
    return np.concatenate([x, np.full((n_pad - len(x),) + x.shape[1:], value, x.dtype)])
  out.update({
      'template_coords': pad(f['template_coords']),
      'element': pad(f['element']),
      'charge': pad(f['charge']),
      'name_encoding': pad(f['name_encoding']),
      'token_idx_at_atom': pad(f['token_idx_at_atom'], -1),
      'ref_space_uid': pad(f['residue_idx'], -1),
      'atom_pad_mask': pad(np.ones(n_atoms, bool), False),
      'atom_mask': pad(np.ones(n_atoms, bool), False),
      'coords_cond': np.zeros((n_pad, 3), np.float32),
  })
  if pad_tokens_to:
    out = pad_tokens(out, pad_tokens_to)
  if not return_topology:
    return out, atom_keys
  topology = {
      'names': [k[2] for k in atom_keys], 'labels': labels, 'element': f['element'],
      'residue_idx': f['residue_idx'], 'mol_type': f['mol_type'], 'residue_type': f['residue_type'],
      'token_idx_at_atom': f['token_idx_at_atom'], 'atom_asym': np.array(tok['asym_id'], np.int64)[
          f['token_idx_at_atom']], 'asym_id': out['asym_id'][:T], 'entity_id': out['entity_id'][:T],
      'bond_index': bond_index, 'ligand_ccd': ligand_ccd, 'n_pad': out['template_coords'].shape[0]}
  return out, atom_keys, topology


# {chain id: Template}: a SMILES ligand's conformer to use instead of a fresh draw
# (the parity harness sets the vendor's own here; a run leaves it empty).
SMILES_TEMPLATES = {}
# The steering potentials' RDKit ligand tables to use instead of building them
# (the harness sets the vendor's: RDKit releases differ in bounds and CIP ranks).
STEERING_LIGAND_TABLES = {}


def pad_atoms(f, n):
  """More atom padding (batch_collate pads only to the window)."""
  g = dict(f)
  extra = n - f['template_coords'].shape[0]
  if extra <= 0:
    return g
  fill = {'token_idx_at_atom': -1, 'ref_space_uid': -1, 'atom_pad_mask': False, 'atom_mask': False}
  for k in ('template_coords', 'element', 'charge', 'name_encoding', 'token_idx_at_atom',
            'ref_space_uid', 'atom_pad_mask', 'atom_mask', 'coords_cond', 'x0'):
    if k not in f:
      continue
    x = f[k]
    g[k] = np.concatenate([x, np.full((extra,) + x.shape[1:], fill.get(k, 0), x.dtype)])
  return g


CONTACT_CLASSES = {'UNSPECIFIED': 0, 'UNSELECTED': 1, 'POCKET>BINDER': 2, 'BINDER>POCKET': 3,
                   'CONTACT': 4}


def load_contacts(path):
  """Contacts from a JSON file, in BioEmu2's Contact form:
  [{"token1": {"asym_id": "A", "res_id": 12}, "token2": {"asym_id": "B",
  "atom_name": "C3"}, "max_distance": 6.0}, ...]. A token's res_id defaults to
  1 only when it names an atom (ContactToken.validate_explicit_res_id)."""
  import json
  with open(path) as fh:
    raw = json.load(fh)
  out = []
  for c in raw:
    extra = set(c) - {'token1', 'token2', 'max_distance'}
    if extra:
      raise ValueError(f'unknown contact field(s) {sorted(extra)}')
    toks = []
    for k in ('token1', 'token2'):
      t = dict(c[k])
      if set(t) - {'asym_id', 'res_id', 'atom_name'}:
        raise ValueError(f'unknown contact token field(s) in {t}')
      if t.get('atom_name') is None and 'res_id' not in t:
        raise ValueError(f'{t}: res_id must be explicitly set when atom_name is None')
      if int(t.get('res_id', 1)) < 1:
        raise ValueError(f'{t}: res_id is 1-based')
      toks.append((t['asym_id'], int(t.get('res_id', 1)), t.get('atom_name')))
    d = float(c.get('max_distance', 6.0))
    if not 4.0 <= d <= 20.0:
      raise ValueError(f'max_distance {d} is outside [4, 20] A, the range BioEmu2 was trained on')
    out.append((toks[0], toks[1], d))
  return out


def _contact_features(contacts, chain_info, f, centre, T):
  """_add_contact_conditioning: every pair UNSELECTED, the named pairs CONTACT
  (both ways) at their distance. A residue token is found by chain and
  chain-local residue; a named atom picks the token centred on that atom; a
  residue with several tokens (a modified residue) is addressed by its CA or
  C1' token."""
  names = [None] * T
  local = np.zeros(T, np.int64)
  chain_of = {}
  for c, (ch, kind, payload, residues, t0, t1) in enumerate(chain_info):
    chain_of[ch.id] = (c, kind, payload, t0, t1)
  res_at_token = np.zeros(T, np.int64)
  res_at_token[f['token_idx_at_atom']] = f['residue_idx']
  atom_names = []
  for row in f['name_encoding']:
    atom_names.append(''.join(chr(x + 32) for x in row if x).strip())
  for t in range(T):
    names[t] = atom_names[centre[t]]
  for ch, (c, kind, payload, t0, t1) in chain_of.items():
    local[t0:t1] = res_at_token[t0:t1] - res_at_token[t0]

  def resolve(tok):
    cid, res_id, atom = tok
    if cid not in chain_of:
      raise ValueError(f'contact token {tok}: no chain {cid}')
    c, kind, payload, t0, t1 = chain_of[cid]
    if kind in ('ccd', 'smiles'):
      if not atom:
        raise ValueError(f'contact token {tok} addresses a ligand and must name an atom')
      if res_id != 1:
        raise ValueError(f'contact token {tok}: a ligand has one residue')
    else:
      if kind == 'branched' and not atom:
        raise ValueError(f'contact token {tok} addresses a branched entity and must name an atom')
      n_res = len(payload)
      if res_id > n_res:
        raise ValueError(f'contact token {tok}: res_id exceeds the chain length {n_res}')
    sel = [t for t in range(t0, t1) if local[t] == res_id - 1]
    if atom:
      sel = [t for t in sel if names[t] == atom]
    elif len(sel) > 1:
      sel = [t for t in sel if names[t] in ('CA', "C1'")]
    if len(sel) != 1:
      raise ValueError(f'contact token {tok} does not match exactly one token')
    return sel[0]

  cls = np.full((T, T), CONTACT_CLASSES['UNSELECTED'], np.int64)
  thr = np.zeros((T, T), np.float32)
  for a, b, d in contacts:
    i, j = resolve(a), resolve(b)
    cls[i, j] = cls[j, i] = CONTACT_CLASSES['CONTACT']
    thr[i, j] = thr[j, i] = d
  return {'contact_conditioning': np.eye(len(CONTACT_CLASSES), dtype=np.int64)[cls],
          'contact_threshold': thr}


def pad_tokens(f, n):
  """Token padding as batch_collate pads a shorter member of a batch."""
  t = f['res_type'].shape[0]
  extra = n - t
  if extra <= 0:
    return f
  def p(x, axes, value=0):
    w = [(0, 0)] * x.ndim
    for a in axes:
      w[a] = (0, extra)
    return np.pad(x, w, constant_values=value)
  g = dict(f)
  for k in ('res_type', 'asym_id', 'entity_id', 'sym_id', 'token_center_atom_index',
            'method_feature', 'pocket_feature', 'deletion_mean'):
    g[k] = p(f[k], [0])
  g['residue_index'] = p(f['residue_index'], [0], -1)
  g['token_pad_mask'] = p(f['token_pad_mask'], [0], False)
  g['profile'] = p(f['profile'], [0])
  g['token_bonds'] = p(f['token_bonds'], [0, 1])
  g['contact_conditioning'] = p(f['contact_conditioning'], [0, 1])
  g['contact_threshold'] = p(f['contact_threshold'], [0, 1])
  for k in ('msa', 'msa_paired', 'deletion_value', 'has_deletion', 'msa_mask'):
    g[k] = p(f[k], [1])
  return g


# --- MSA (msa/colabfold.tokenize_msa + msa/features.process_msa_features) ---------------

def _a3m_rows(a3m):
  return [l.strip() for l in a3m.strip().splitlines()
          if l.strip() and not l.startswith('>') and not l.startswith('#')]


def _query_of(row):
  return ''.join(c for c in row if not c.islower() and c not in '-.').upper()


def _encode_rows(rows):
  """_build_msa_from_colabfold_seqs: (residue tokens, {res_idx: deletions}) per row."""
  out = []
  for row in rows:
    res, dels, count = [], {}, 0
    for c in row:
      if c != '-' and c.islower():
        count += 1
        continue
      if count:
        dels[len(res)] = count
        count = 0
      res.append(_PROT[c])
    out.append((res, dels))
  return out


def _msa_features(chain_order, chain_info, f, centre, T):
  protein = [i for i, (_, kind, *_r) in enumerate(chain_info) if kind == 'protein']
  seqs = {i: chain_info[i][2] for i in protein}

  def unpaired_rows(i):
    a3m = chain_info[i][0].unpaired_msa
    rows = _a3m_rows(a3m) if a3m else []
    if not rows:
      rows = [seqs[i]]
    if _query_of(rows[0]) != seqs[i].upper():
      raise ValueError(f'MSA query of chain {chain_info[i][0].id} is not its sequence')
    return rows

  heteromeric = len(set(seqs.values())) > 1
  paired_given = heteromeric and all(chain_info[i][0].paired_msa for i in protein)
  n_chains = len(chain_info)
  msas, pairing, is_paired = {}, [], []
  if paired_given:
    paired_rows = {i: [r for r in _a3m_rows(chain_info[i][0].paired_msa)[:MAX_PAIRED_SEQS + 1][1:]
                       if r.replace('-', '')][:MAX_PAIRED_SEQS] for i in protein}
    unpaired = {i: unpaired_rows(i)[:MAX_SEQS] for i in protein}
    depth = min(min(len(r) for r in paired_rows.values()), MAX_SEQS - 1)
    for i in protein:
      combined = [seqs[i]] + paired_rows[i][:depth]
      start = 1 if unpaired[i] and unpaired[i][0].replace('-', '').upper() == seqs[i].upper() else 0
      combined += unpaired[i][start:start + MAX_SEQS - len(combined)]
      msas[i] = _encode_rows(combined)
    pairing.append({c: 0 for c in range(n_chains)})
    is_paired.append({c: 1 for c in range(n_chains)})
    for k in range(1, depth + 1):
      pairing.append({c: (k if c in protein else -1) for c in range(n_chains)})
      is_paired.append({c: (1 if c in protein else 0) for c in range(n_chains)})
    offset = depth + 1
    max_unpaired = max(len(msas[c]) - offset for c in protein)
    for k in range(max_unpaired):
      pairing.append({c: ((offset + k) if c in protein and offset + k < len(msas[c]) else -1)
                      for c in range(n_chains)})
      is_paired.append({c: 0 for c in range(n_chains)})
      if len(pairing) >= MAX_SEQS:
        break
  else:
    for i in protein:
      msas[i] = _encode_rows(unpaired_rows(i)[:MAX_SEQS])
    homomeric = len(set(seqs.values())) == 1 and len(protein) > 1
    n_seqs = min(min((len(msas[c]) for c in protein), default=1), MAX_SEQS)
    for k in range(n_seqs):
      flag = 1 if homomeric else (1 if k == 0 else 0)
      is_paired.append({c: flag for c in range(n_chains)})
      pairing.append({c: (k if c in protein else (0 if k == 0 else -1)) for c in range(n_chains)})
  # dummy MSAs: the chain's own residue types, one row
  for c in range(n_chains):
    if c not in msas:
      ch, kind, payload, residues, t0, t1 = chain_info[c]
      msas[c] = [([r.restype for r in residues], {})]

  # token -> (chain, chain-local residue)
  res_at_token = np.zeros(T, np.int64)
  res_at_token[f['token_idx_at_atom']] = f['residue_idx']
  asym = np.zeros(T, np.int64)
  for c, info in enumerate(chain_info):
    asym[info[4]:info[5]] = c
  local = res_at_token.copy()
  for c, info in enumerate(chain_info):
    local[info[4]:info[5]] -= res_at_token[info[4]]

  S = len(pairing)
  msa = np.full((S, T), GAP, np.int64)
  dele = np.zeros((S, T), np.float32)
  paired = np.zeros((S, T), np.int64)
  for t in range(T):
    c, r = asym[t], local[t]
    for s in range(S):
      paired[s, t] = is_paired[s][c]
      k = pairing[s][c]
      if k != -1:
        res, dels = msas[c][k]
        msa[s, t] = res[r]
        dele[s, t] = dels.get(r, 0)
  onehot = np.eye(NUM_TOKENS, dtype=np.int64)[msa]
  value = (math.pi / 2 * np.arctan(dele / 3)).astype(np.float32)
  return {
      'msa': onehot,
      'msa_paired': paired,
      'deletion_value': value,
      'has_deletion': dele > 0,
      'deletion_mean': value.mean(0),
      'profile': onehot.astype(np.float32).mean(0),
      'msa_mask': np.ones((S, T), np.int64),
  }
