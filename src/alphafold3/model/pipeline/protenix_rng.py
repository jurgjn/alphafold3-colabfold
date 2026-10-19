"""Protenix's inference-time randomness, replayed from the seed as native draws it (torch 2.13, one CUDA device).

Native Protenix seeds Python, NumPy and torch (CPU and CUDA) per seed, then at inference
  * decides once per run whether to apply MC dropout (p=0.4) to the recycled pair: `random.random() < 0.4`,
    after its polymer builder spent a `random.sample(leaving_groups, bonds)` per bonded atom, once per entity;
  * samples the MSA rows every pass: a size from `torch.randint` on the CPU generator (after the
    DataLoader's int64 base seed) and the rows from `torch.randperm` on the CUDA generator;
  * drops the recycled pair with torch's fused dropout kernel on the CUDA generator.
The CUDA draws are cuRAND Philox4_32_10 at torch's (seed, offset); which draw lands on which element depends on
the launch grid, which torch sizes from the device's SM count, so a run replays native on the same GPU model.
"""
import random

import numpy as np

from alphafold3.model.network import torch_philox

_M32 = np.uint64(0xFFFFFFFF)
BLOCK = 256
MC_DROPOUT_RATE = 0.4
MC_APPLY_RATE = 0.4


def philox(seed, subseq, block):
  """cuRAND Philox4_32_10: (..., 4) uint32 for key=seed and counter=(block, subseq); the TF layout of random_philox."""
  subseq, block = np.broadcast_arrays(np.asarray(subseq, np.uint64), np.asarray(block, np.uint64))
  halves = lambda x: ((x & _M32).astype(np.uint32), (x >> np.uint64(32)).astype(np.uint32))
  key = halves(np.full(block.shape, seed, np.uint64))
  with np.errstate(over='ignore'):
    return np.stack(torch_philox.philox4x32(key, halves(block) + halves(subseq)), -1)


def grid_size(nelem, sms, threads_per_sm):
  return min(sms * (threads_per_sm // BLOCK), (nelem + BLOCK - 1) // BLOCK)


def dropout_offset_increment(nelem, sms, threads_per_sm):
  """philox offset consumed by torch's fused dropout on nelem elements"""
  return ((nelem - 1) // (BLOCK * grid_size(nelem, sms, threads_per_sm) * 4) + 1) * 4


def _round4(n):
  return (n + 3) // 4 * 4


def randperm(seed, offset, n, sms, threads_per_sm):
  """torch.randperm(n) on CUDA (32-bit key path) and the philox offset it consumes"""
  nd = float(n)
  bits = min(64, int(np.ceil(np.log2(nd - (6 * nd * nd + 1) / (np.log(0.9) * 12)))))
  if bits > 32:
    raise NotImplementedError(f'randperm of {n} needs 64-bit keys')
  # random_(int32 min, int32 max): 64-bit path, two values per curand4, grid-stride over the elements
  grid = grid_size(n, sms, threads_per_sm)
  span = BLOCK * grid
  e = np.arange(n)
  it, within = e // (2 * span), e % (2 * span)
  t, half = within % span, within // span
  u = philox(seed, t, offset // 4 + it)
  hi = np.where(half == 0, u[:, 0], u[:, 2]).astype(np.uint64)
  lo = np.where(half == 0, u[:, 1], u[:, 3]).astype(np.uint64)
  keys = (((hi << np.uint64(32)) | lo) % np.uint64(2 ** 32 - 1)).astype(np.int64) - 2 ** 31
  used = ((n - 1) // (2 * span) + 1) * 4
  mask = (1 << bits) - 1
  order = np.argsort(keys & mask, kind='stable')
  data, sk = order.copy(), keys[order] & mask
  off2 = offset + used
  for tid in np.flatnonzero((sk[:-1] == sk[1:]) & np.r_[True, sk[1:-1] != sk[:-2]]):
    size = 1
    while tid + size < n and sk[tid + size] == sk[tid]:
      size += 1
    stream = philox(seed, tid, off2 // 4 + np.arange((size + 2) // 4 + 1)).reshape(-1)
    seg = data[tid:tid + size]
    for c, i in enumerate(range(size - 1, 0, -1)):
      r = int(stream[c]) % (i + 1)
      seg[i], seg[r] = seg[r], seg[i]
  return data, used + _round4(n)


def _mol_type(comp):
  """ccd.get_mol_type: protein, dna, rna or ligand from _chem_comp.type"""
  t = comp['_chem_comp.type'][0].upper() if comp else ''
  if 'PEPTIDE' in t and t != 'PEPTIDE-LIKE':
    return 'protein'
  return 'dna' if 'DNA' in t else 'rna' if 'RNA' in t else 'ligand'


def _leaving_groups(comp):
  """ccd._map_central_to_leaving_groups on the heavy atoms: central atom -> number of leaving groups, or None"""
  heavy = [e.upper() not in ('H', 'D') for e in comp['_chem_comp_atom.type_symbol']]
  names = [n for n, h in zip(comp['_chem_comp_atom.atom_id'], heavy) if h]
  leaving = {n for n, f, h in zip(comp['_chem_comp_atom.atom_id'], comp['_chem_comp_atom.pdbx_leaving_atom_flag'], heavy)
             if h and f == 'Y'}
  adj = {n: [] for n in names}
  for a, b in zip(comp.get('_chem_comp_bond.atom_id_1', ()), comp.get('_chem_comp_bond.atom_id_2', ())):
    if a in adj and b in adj:
      adj[a].append(b)
      adj[b].append(a)
  groups = {}
  for c in names:
    if c in leaving:
      continue
    for l in [x for x in adj[c] if x in leaving]:
      adj[c].remove(l)
      adj[l].remove(c)
      seen, todo = {l}, [l]
      while todo:
        for y in adj[todo.pop()]:
          if y not in seen:
            seen.add(y)
            todo.append(y)
      if seen - leaving:
        return None
      groups[c] = groups.get(c, 0) + 1
  return groups


def leaving_group_draws(entities, ccd):
  """(leaving groups, bonds) of each random.sample native's polymer builder makes, in its order

  entities are the CCD code lists of the polymer entities (one per distinct chain, not per copy); ccd maps a code to its
  mmCIF dict. Native bonds C-N between protein and O3'-P between nucleic residues and samples at both ends.
  """
  info = {}
  for code in {c for codes in entities for c in codes}:
    comp = ccd.get(code)
    heavy = set() if comp is None else {n for n, e in zip(comp['_chem_comp_atom.atom_id'], comp['_chem_comp_atom.type_symbol'])
                                        if e.upper() not in ('H', 'D')}
    info[code] = (_mol_type(comp), (_leaving_groups(comp) if comp else None) or {}, heavy)
  draws = []
  for codes in entities:
    bonds = {}
    for k in range(len(codes) - 1):
      (t1, _, a1), (t2, _, a2) = info[codes[k]], info[codes[k + 1]]
      if t1 == t2 == 'protein':
        x, y = 'C', 'N'
      elif t1 in ('dna', 'rna') and t2 in ('dna', 'rna'):
        x, y = "O3'", 'P'
      else:
        continue
      if x in a1 and y in a2:
        bonds[k, x] = bonds.get((k, x), 0) + 1
        bonds[k + 1, y] = bonds.get((k + 1, y), 0) + 1
    draws += [(info[codes[k]][1][a], b) for (k, a), b in bonds.items() if info[codes[k]][1].get(a)]
  return draws


def _randbelow(r, n):
  """Random._randbelow_with_getrandbits, spelled out so the draw count does not depend on the Python version"""
  k = n.bit_length()
  while r.getrandbits(k) >= n:
    pass


def mc_dropout_decision(seed, draws):
  """native's once-per-run MC dropout draw, after the leaving-group samples (leaving_group_draws)"""
  r = random.Random(seed)
  for n, b in draws:
    if b <= n:
      for i in range(b):
        _randbelow(r, n - i)
  return r.random() < MC_APPLY_RATE


def msa_sample_sizes(seed, n_msa, passes):
  """torch.randint(1, n + 1) per pass on the CPU generator, after the DataLoader's int64 base seed"""
  bg = np.random.MT19937()
  bg._legacy_seeding(seed)
  raw = [int(bg.random_raw()) for _ in range(2 + passes)]
  return [r % n_msa + 1 for r in raw[2:]]


def replay(seed, *, leaving_draws, n_msa, n_tokens, pair_channel, passes, sms, threads_per_sm):
  """per-pass draws: mc_dropout, MSA row subsets (as index lists) and the dropout philox offsets"""
  mc = mc_dropout_decision(seed, leaving_draws)
  sizes = msa_sample_sizes(seed, n_msa, passes)
  nelem = n_tokens * n_tokens * pair_channel
  offset, rows, drop_offsets = 0, [], []
  for k in range(passes):
    drop_offsets.append(offset)
    if mc:
      offset += dropout_offset_increment(nelem, sms, threads_per_sm)
    perm, used = randperm(seed, offset, n_msa, sms, threads_per_sm)
    offset += used
    rows.append(perm[:sizes[k]])
  return mc, rows, drop_offsets
