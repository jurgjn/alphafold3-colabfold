# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Per-model featurisation conventions, applied to a featurised batch.

`global_config.model` decides which forward branches run; a few families also
need the INPUT built their way. Those differences are conventions their weights
were trained under, not preferences -- getting one wrong is silent, and shows up
as a fold that is merely mediocre.

Applied on the finished batch rather than threaded through the featuriser: each
one rewrites a handful of plain index arrays, so the layout machinery upstream
stays untouched and stock AlphaFold 3 is byte-identical to before.

    batch = model_features.apply(batch, model_registry.get('opendde'), ...)
"""

from __future__ import annotations

import os

import numpy as np


def _key_window(batch, policy, k_size=None, prefix=''):
  """Rewrite the atom-attention key window on a FINISHED batch.

  Three models needed three windows and got three near-identical functions. The
  arithmetic is the same in all of them -- each block of `q_size` queries takes
  `k_size` keys centred on it, starting at `block_start - (k_size - q_size)//2`,
  and the same four gather arrays are rewritten -- so what actually differs is
  the END POLICY, and for ESMFold2 the width:

    'slide'  AF3's own (features.py AtomCrossAtt, "Shift subsets with
             out-of-bound indices"): shift an out-of-range window bodily back
             inside the real atom count, so every block keeps k_size valid keys.
    'wrap'   chai-1: the window modulo the padded atom count (`% num_atoms` in
             its traced graph), so the first block's leading keys land in
             padding and are masked out -- its end blocks genuinely see fewer
             real neighbours.
    'pad'    opendde/protenix (rearrange_qk_to_dense_trunk): clip and mask, so
             edge blocks see fewer neighbours AND the ones they see sit at
             DIFFERENT key slots, which misaligns the per-slot pair bias too.
    'slide_qblock'
             IntelliFold-2: 'slide', but the edge is the atom count ROUNDED UP
             to a whole query block, not the real atom count. if2 reshapes its
             flat atom axis into windows (`b (n w) -> b n w`), so it must first
             pad the axis to a multiple of 32 -- and its edge windows are then
             aligned to THAT length. On 6MRR (574 real atoms) its last two
             blocks take keys 448..575, where AF3's slide gives 446..573: two
             extra REAL keys on our side, because 574 and 575 are padding for
             if2 and masked. Two atoms, and it is the whole of if2's atom
             encoder residual -- see the localisation in the commit message.

  Rewritten on the finished batch rather than threaded through AF3's
  featurisation: the key gathers are plain index arrays over the flat queries
  layout, so the window is the only thing that has to change and the rest of the
  layout machinery stays untouched.
  """
  q_mask = np.asarray(batch[f'{prefix}token_atoms_to_queries:gather_mask'])
  num_subsets, q_size = np.asarray(
      batch[f'{prefix}token_atoms_to_queries:gather_idxs']).shape
  n_padded = num_subsets * q_size
  kk = f'{prefix}queries_to_keys:gather_idxs'
  if k_size is None:
    k_size = np.asarray(batch[kk]).shape[1]
  else:
    # only when a width is REQUESTED: a tiny input cannot hold it, and the
    # window is then the whole list. Taking this on the width read back from the
    # batch would silently reshape arrays the other two policies keep.
    k_size = min(k_size, n_padded)
  starts = np.arange(num_subsets) * q_size + (q_size - k_size) // 2
  flat_mask = q_mask.reshape(-1)
  if policy == 'slide':
    starts = np.clip(starts, 0, n_padded - k_size)
  elif policy == 'slide_qblock':
    # The edge is the REAL atom count rounded up to a whole query block. Not
    # `n_padded` (that is num_tokens * max_atoms, far past the real atoms, so
    # nothing would ever slide) and not the real count itself (that is AF3's
    # own convention, which features.py already applies).
    n_real = int(flat_mask.sum())
    bound = -(-n_real // q_size) * q_size
    starts = np.clip(starts, 0, max(bound - k_size, 0))
  window = starts[:, None] + np.arange(k_size)[None, :]

  if policy == 'wrap':
    window = window % n_padded
    keep = flat_mask[window]
  elif policy == 'pad':
    in_bounds = (window >= 0) & (window < n_padded)
    window = np.clip(window, 0, n_padded - 1)
    keep = in_bounds & flat_mask[window]
  elif policy in ('slide', 'slide_qblock'):
    keep = flat_mask[window]
  else:
    raise ValueError(f'unknown key-window policy {policy!r}')

  batch[kk] = window.astype(np.asarray(batch[kk]).dtype)
  batch[f'{prefix}queries_to_keys:gather_mask'] = keep
  tok = np.asarray(batch[f'{prefix}tokens_to_queries:gather_idxs']).reshape(-1)
  tk = f'{prefix}tokens_to_keys:gather_idxs'
  batch[tk] = tok[window].astype(np.asarray(batch[tk]).dtype)
  tok_mask = np.asarray(
      batch[f'{prefix}tokens_to_queries:gather_mask']).reshape(-1)[window]
  batch[f'{prefix}tokens_to_keys:gather_mask'] = (
      keep if policy == 'pad' else tok_mask)
  return batch


def _circular_key_window(batch, prefix=''):
  """chai-1's atom-attention key window wraps; AF3's is shifted in-bounds.

  Both models give each block of 32 queries a 128-key window centred on it, and
  both compute the same window: block_start - 48 .. block_start + 79. They only
  differ at the ends. AF3 slides an out-of-range window bodily back inside the
  real atom count, so every block keeps 128 valid keys. chai takes the window
  modulo the padded atom count (`% num_atoms` in its traced graph), so the first
  block's leading 48 keys land in padding and are masked out -- the end blocks
  genuinely see fewer real neighbours.

  Rewritten on the finished batch rather than threaded through AF3's
  featurisation: the three key gathers are plain index arrays over the flat
  queries layout, so the window is the only thing that has to change and the
  rest of the layout machinery stays untouched.
  """
  return _key_window(batch, 'wrap', prefix=prefix)

def _padded_key_window(batch, prefix=''):
  """opendde/protenix zero-PAD the atom-attention key window; AF3 slides it.

  Both give each block of 32 queries a 128-key window centred on it:
  block_start - 48 .. block_start + 79. AF3 then shifts any window that runs off
  either end bodily back inside the real atom count, so every block keeps 128
  valid keys (features.py AtomCrossAtt: "Shift subsets with out-of-bound
  indices"). opendde's rearrange_qk_to_dense_trunk instead pads the key sequence
  with (n_keys - n_queries) // 2 = 48 zeros on the left and the matching amount
  on the right, and masks those slots -- its edge blocks genuinely see fewer
  neighbours, and the neighbours they do see sit at DIFFERENT key slots than
  ours, so the per-slot pair bias is misaligned too.

  Verified against native's own bookkeeping: for 1EHZ its pad_info reads
  k_pad_left 48 / k_pad_right 54, and its v_lm disagreed with ours on 2.5% of
  window entries -- all of them in blocks 0, 1 and the last two, which is exactly
  the set of blocks where sliding and padding differ.

  Only the four edge blocks change, which is a small share of a long chain but a
  large one of a short chain: 4 of 51 query blocks for 1EHZ's 1626 atoms against
  4 of 18 for 6MRR's 574.
  """
  return _key_window(batch, 'pad', prefix=prefix)

def _drop_atoms_by_name(batch, names):
  """Mask out atoms another model's tokenizer does not create.

  AF3 gives the C-terminal residue an OXT; chai-1 does not -- its 6MRR run has
  573 atoms where ours has 574, and the extra one is exactly that OXT (verified
  against chai's own AtomRefPos/AtomNameOneHot dump). One atom in 574 sounds
  ignorable, but it lands in the last token's pooled atom representation and in
  the atom-pair window around it, and the trunk amplifies s_init error several
  fold per pass.

  The atom layout is COMPACTED (real atoms first, 18 full blocks of 32 here), so
  dropping an atom that is not last would renumber every atom after it. This
  only masks: the dense slot keeps its coordinates and simply stops being real,
  which is what every consumer already keys off.
  """
  chars = np.asarray(batch['ref_atom_name_chars'])
  decode = lambda row: ''.join(
      chr(int(c) + 32) if 0 <= c < 64 else '' for c in row).strip()
  ref_mask = np.array(batch['ref_mask'])
  n_atoms_per_token = ref_mask.shape[1]
  # STANDARD RESIDUES ONLY. Every vendor that drops these drops them from
  # standard residues and keeps them elsewhere, and says so: of3's
  # `remove_std_residue_terminal_atoms` is named for it ("terminal atoms can be
  # kept for any non-standard residues, as they are tokenized per-atom") and
  # rf3's filter carries `& ~is_atomized`.
  #
  # It is not pedantry, because these names mean different things in different
  # residues. **O3P is a sidechain atom of phosphoserine.** Dropping by name
  # alone took our SEP-20 ubiquitin to 604 atoms where of3's own featuriser has
  # 605 -- we removed the OXT correctly and then removed a phosphate oxygen that
  # native keeps. An ATOMISED residue is one token per atom, so it is identified
  # here by its token holding exactly one real atom.
  # `n_real > 1` alone identifies an atomised residue ONLY under AF3's
  # convention, where each of its atoms is its own token. boltz2 keeps a
  # modified residue as ONE token holding every atom
  # (`modified_as_one_token`), so that test called phosphoserine a standard
  # residue and dropped its O3P -- the hazard this comment block already
  # warned about, reintroduced through the other knob. `is_modified` is set by
  # `_mark_modified_residues`, which apply() runs BEFORE this, so the flag is
  # the reliable test and the atom count is the fallback for models that do not
  # set it.
  #
  # Measured on ACSEFGHIKLWY with SEP at 3, boltz2: dropping O3P leaves the
  # phosphate under-coordinated and collapses P-O1P and P-O2P to ~0.5 of ideal
  # (mean bond ratio 0.880, rms 0.395 A). Keeping it reads 0.987 at rms 0.080,
  # which is genuine Boltz-2's own 0.986-1.011 at 0.043-0.076.
  n_real = ref_mask.sum(axis=1)
  modified = np.asarray(batch.get('is_modified',
                                  np.zeros(ref_mask.shape[0], bool))
                        ).astype(bool).reshape(-1)
  drop = {(t, a) for t in range(ref_mask.shape[0])
          for a in range(n_atoms_per_token)
          if ref_mask[t, a] and decode(chars[t, a]) in names
          and n_real[t] > 1 and not modified[t]}
  return _mask_atoms(batch, drop)


def _mask_atoms(batch, drop):
  """Mask (token, atom slot) pairs out of every atom gather and compact the layouts; see _drop_atoms_by_name."""
  if not drop:
    return 0
  ref_mask = np.array(batch['ref_mask'])
  n_atoms_per_token = ref_mask.shape[1]
  flat = {t * n_atoms_per_token + a for t, a in drop}
  for t, a in drop:
    ref_mask[t, a] = False
  batch['ref_mask'] = ref_mask
  if 'pred_dense_atom_mask' in batch:
    pdm = np.array(batch['pred_dense_atom_mask'])
    for t, a in drop:
      pdm[t, a] = False
    batch['pred_dense_atom_mask'] = pdm

  # queries carry the dropped atoms too; clear them before anything gathered
  # THROUGH the queries (the key window) is rebuilt from them.
  q_idxs = np.asarray(batch['token_atoms_to_queries:gather_idxs'])
  q_mask = np.array(batch['token_atoms_to_queries:gather_mask'])
  dead_queries = set()
  for r, c in zip(*np.nonzero(q_mask)):
    if int(q_idxs[r, c]) in flat:
      q_mask[r, c] = False
      dead_queries.add(r * q_idxs.shape[1] + c)
  batch['token_atoms_to_queries:gather_mask'] = q_mask

  for key, dead in (('queries_to_token_atoms', flat),
                    ('token_atoms_to_pseudo_beta', flat),
                    ('queries_to_keys', dead_queries)):
    idxs = np.asarray(batch[f'{key}:gather_idxs'])
    mask = np.array(batch[f'{key}:gather_mask'])
    if key == 'queries_to_token_atoms':
      # this one is indexed BY (token, atom), so drop by position
      for t, a in drop:
        mask[t, a] = False
    else:
      mask &= ~np.isin(idxs, list(dead))
    batch[f'{key}:gather_mask'] = mask

  _remove_dropped_atoms_from_layouts(batch, drop)
  return len(drop)


def _remove_dropped_atoms_from_layouts(batch, drop):
  '''REMOVE the dropped atoms from the layouts, the way the vendors do.

  Masking a finished batch is not the same thing as the vendor never emitting
  the atom, because a mask leaves a HOLE in the flat atom axis and two things
  are cut on that axis:

    * the 32-atom attention windows. Every block after the hole has a different
      membership than the vendor's compacted axis. This is the whole of the
      openfold3/openbind0 atom-encoder regression of 2026-09-12 (14 BAD cells
      that read corr 1.000000 two days earlier, with the gate headers reading
      `574 real atoms` before and `573` after). rf3, if2 and boltz2 took the
      same drop and stayed exact only because their native adapters consume our
      flat features hole and all, so it cancels on both sides.
    * the mmCIF gather, which reaches users. `flat_output_layout` and
      `empty_output_struc` are built during featurisation and the drop never
      touched them, so the model stopped predicting an atom the output still
      asked for -- and `predicted_structure_from_coords` writes a missing atom
      at (0, 0, 0) behind a `logging.warning`. openfold3 and boltz2 were handing
      back 574 output atoms for 573 predicted ones.

  `token_atoms_mask` is just `atom_name.astype(bool)`, so blanking the name
  compacts the flat axis by itself; the AtomCrossAtt gathers are then rebuilt
  from the compacted layout and the output layout is filtered to match.
  '''
  import dataclasses

  from alphafold3.model import features as _features

  lay = _features._unwrap(batch.get('token_atoms_layout'))
  if lay is None:
    return

  names = np.array(lay.atom_name, copy=True)
  gone = [(str(lay.chain_id[t, a]), int(lay.res_id[t, a]), str(names[t, a]))
          for t, a in drop if names[t, a]]
  for t, a in drop:
    names[t, a] = ''
  lay = dataclasses.replace(lay, atom_name=names)
  batch['token_atoms_layout'] = np.array(lay, object)

  # Rebuild the atom cross-attention gathers on the COMPACTED axis. The subset
  # sizes and the padded atom count are read back off the batch rather than
  # re-derived from config, so this cannot drift from the arrays it replaces.
  q_idxs = np.asarray(batch['token_atoms_to_queries:gather_idxs'])
  num_subsets, q_size = q_idxs.shape
  k_size = np.asarray(batch['queries_to_keys:gather_idxs']).shape[1]
  shapes = _features.PaddingShapes(
      num_tokens=lay.shape[0], msa_size=0, num_chains=0, num_templates=0,
      num_atoms=num_subsets * q_size)
  # `flat_atom_order` ranks the rows of the OLD flat axis, so removing rows
  # invalidates it. No model needs both today -- opendde is the only one with a
  # structural atom order and it KEEPS its terminal atoms (918/1204 against its
  # own featuriser) -- so this refuses the combination rather than silently
  # dropping a permutation that reorders every atom.
  if batch.get('flat_atom_order') is not None:
    raise NotImplementedError(
        'dropping atoms from a model that also permutes the flat atom axis '
        '(flat_atom_order) would need the permutation re-ranked over the '
        'remaining atoms; no model currently does both')
  aca = _features.AtomCrossAtt.compute_features(
      all_token_atoms_layout=lay,
      queries_subset_size=q_size,
      keys_subset_size=k_size,
      padding_shapes=shapes)
  batch.update(aca.as_data_dict())

  # The OUTPUT side: drop the same atoms from the flat output layout and from
  # the empty structure the coordinates are written into, so nothing is left
  # asking for an atom the model no longer predicts.
  fol = _features._unwrap(batch.get('flat_output_layout'))
  struc = _features._unwrap(batch.get('empty_output_struc'))
  if fol is None or not gone:
    return
  keep = np.ones(fol.shape[0], bool)
  gone_set = set(gone)
  for i in range(fol.shape[0]):
    if (str(fol.chain_id[i]), int(fol.res_id[i]),
        str(fol.atom_name[i])) in gone_set:
      keep[i] = False
  if keep.all():
    return
  batch['flat_output_layout'] = np.array(fol[keep], object)
  if struc is not None:
    # empty_output_struc is indexed 1:1 by flat_output_layout, so the SAME
    # boolean is applied to both. Filtering by atom NAME instead would also
    # remove atoms that were deliberately kept -- an OXT inside an atomised
    # residue, which is exactly the distinction the drop itself is careful
    # about.
    if struc.num_atoms != keep.shape[0]:
      raise ValueError(
          'empty_output_struc has %d atoms, flat_output_layout %d -- they are '
          'meant to be 1:1' % (struc.num_atoms, keep.shape[0]))
    batch['empty_output_struc'] = np.array(struc.filter(mask=keep), object)




def _override_ref_conformers(batch, conformers, tokens=None):
  """Replace ref_pos with another model's reference conformers, by atom name.

  AF3 (and so featurise_spec) takes reference coordinates from the CCD ideal
  values. chai-1 instead ships a pre-generated RDKit cache
  (`conformers_v1.apkl`), and the difference is not cosmetic: rebuilding chai's
  ATOM feature stream from our conformers lands at 40% relative error, while the
  same code with chai's own positions is exact to the bfloat16 floor. The
  feature goes straight into a Linear, so it is not pose-invariant.

  Worth knowing that this asymmetry is deliberate on chai's side and applies to
  POLYMERS ONLY: its tokenizer returns the cached conformer unchanged for
  standard residues but calls `center_random_augment()` -- a random rotation
  plus a random translation -- for ligands and non-standard residues. So a
  ligand's reference pose is fresh noise on every run by design, and only the
  polymer conformers have to be matched.

  `conformers` maps a residue code to (atom_names, positions). Atoms whose names
  are absent (AF3's terminal OXT, which chai's conformers do not carry) keep the
  coordinates they already had.
  """
  from alphafold3.constants import residue_names

  pos = np.array(batch['ref_pos'])
  mask = np.asarray(batch['ref_mask'])
  chars = np.asarray(batch['ref_atom_name_chars'])
  aatype = np.asarray(batch['aatype'])
  decode = lambda row: ''.join(
      chr(int(c) + 32) if 0 <= c < 64 else '' for c in row).strip()

  names_3 = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP

  # NEVER SUBSTITUTE INTO AN ATOMISED RESIDUE. AF3 atomises a modified residue
  # into one token per atom and gives every one of them the PARENT restype -- so
  # a phospho-serine's ten tokens all say SER, and a table keyed on restype
  # happily rewrites the six atoms SER has (N, CA, CB, OG, C, O) while the
  # phosphate (P, O1P, O2P, O3P) keeps its CCD coordinates. The result is one
  # conformer stitched from two frames that do not align: measured on SEP, the
  # OG-P bond in ref_pos comes out at 0.43 A against a 1.61 ideal -- a ratio of
  # 0.268 -- while CB-OG, whose atoms come from the same side of the seam, is
  # fine. The model is then asked to place a group whose reference geometry is
  # already broken.
  #
  # An atomised residue is exactly "more than one token sharing (asym_id,
  # residue_index)", which is the convention that defines it.
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi_all = np.asarray(batch['residue_index']).reshape(-1)
  counts = {}
  for key in zip(asym.tolist(), resi_all.tolist()):
    counts[key] = counts.get(key, 0) + 1
  atomised = np.array([counts[k] > 1 for k in zip(asym.tolist(),
                                                  resi_all.tolist())])

  n_replaced = 0
  for t in range(pos.shape[0]) if tokens is None else tokens:
    if atomised[t]:
      continue
    code = names_3[int(aatype[t])] if int(aatype[t]) < len(names_3) else None
    entry = conformers.get(code)
    if entry is None:
      continue
    ref_names, ref_pos = entry
    lookup = {n: i for i, n in enumerate(ref_names)}
    for a in range(pos.shape[1]):
      if not mask[t, a]:
        continue
      j = lookup.get(decode(chars[t, a]))
      if j is not None:
        pos[t, a] = ref_pos[j]
        n_replaced += 1
  batch['ref_pos'] = pos.astype(np.asarray(batch['ref_pos']).dtype)
  return n_replaced


def _raw_last_conformer(batch):
  """boltz2 never centres or rotates its last conformer group: give it boltz's stored conformer, raw."""
  from alphafold3.constants import boltz2_conformers
  uid = np.asarray(batch['ref_space_uid'])
  mask = np.asarray(batch['ref_mask']) > 0
  if not mask.any():
    return 0
  last = int(uid[mask].max())
  tokens = np.nonzero(((uid == last) & mask).any(-1))[0].tolist()
  return _override_ref_conformers(batch, boltz2_conformers.as_conformers(), tokens=tokens)


def _zero_msa(batch):
  """chai-1's single-sequence path: an all-gap MSA, profile and deletion zero.

  Zeroing the PROFILE too, not just the mask: chai's no-MSA path feeds an
  all-gap MSA whose profile and deletion mean are identically zero, whereas AF3's
  featuriser would still hold the query's own profile. Leaving it in adds a term
  chai does not have. (Checked against chai's own captured feature stream from a
  no-MSA run: every row is the GAP class with profile and deletion_mean zero.)
  """
  for key in ('msa_mask', 'profile', 'deletion_mean'):
    batch[key] = np.zeros_like(np.asarray(batch[key]))
  return batch


def _chai_tokenise(rows):
  """a3m rows -> (tokens, deletions) as chai's a3m tokeniser makes them: capitals and '-' are columns in our protein
  classes (other capitals X), lower case counts as a deletion capped at 255, anything else is skipped."""
  from alphafold3.constants import residue_names
  lut = np.full(256, 20, np.int64)
  for c, i in residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP_TO_INT.items():
    lut[ord(c)] = i
  width = sum(c.isupper() or c == '-' for c in rows[0])
  tok = np.full((len(rows), width), 21, np.int64)
  dele = np.zeros((len(rows), width), np.int64)
  for i, row in enumerate(rows):
    b = np.frombuffer(row.encode(), np.uint8)
    col = ((b >= 65) & (b <= 90)) | (b == 45)
    low = np.cumsum((b >= 97) & (b <= 122))
    at = np.flatnonzero(col)[:width]
    ins = low[at]
    tok[i, :at.size] = lut[b[at]]
    dele[i, :at.size] = np.minimum(np.diff(ins, prepend=0), 255)
  return tok, dele


_CHAI_UNIREF_QUOTA = 9999   # msa_dataset_source_to_quota[UNIREF90] 10_000, minus the query


def _chai_msa(batch, fold_input):
  """chai-1's MSA inputs where AF3's differ, once any chain has an alignment.

  A protein chain with an alignment: chai counts the classes of its MSA rows, deduplicated by their tokens, in the
  MSA's uint8 dtype, so each count wraps at 256 before it is normalised by their sum (MSAProfileGenerator's
  scatter_add); its deletion mean is over the same rows. AF3's profile is the true frequency over the undeduplicated
  unpaired rows: on mono825's 10128-row MSA the two correlate 0.44, and native's in place of ours takes s_init from
  corr 0.952 to 0.970 against native's.

  Every other polymer chain is chai's single-sequence context: its sequence string (a modification written "(CCD)")
  through the protein a3m tokeniser, so a DNA A/C/G/T reads ALA/CYS/GLY/THR, as the query row and as a one-hot profile,
  and no other row (masked). AF3 gives it its own classes and real gap rows. Ligands keep create_msa_feat's handling.
  Token-identical rows are then masked, as chai drops them.
  """
  import string
  from alphafold3.common import folding_input as fi
  from alphafold3.constants import residue_names
  from alphafold3.data import parsers

  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  profile = np.array(batch['profile'], np.float32)
  del_mean = np.array(batch['deletion_mean'], np.float32)
  msa, mask, dele = np.array(batch['msa']), np.array(batch['msa_mask']), np.array(batch['deletion_matrix'])
  # chai's colabfold reader: a UniRef* header is UNIREF90 (2), anything else BFD_UNICLUST (0); matched to our rows by
  # their residues on the chain's columns, first a3m occurrence wins
  source = np.full(msa.shape[0], 2, np.int32)
  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  to_cls = {c: vocab.index(residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(c, residue_names.UNK)) for c in string.ascii_uppercase}
  to_cls['-'] = vocab.index(residue_names.GAP)
  for index, chain in enumerate(fold_input.chains):
    sel = np.flatnonzero(asym == index + 1)
    if isinstance(chain, fi.Ligand) or not sel.size:
      continue
    col = resi[sel] - 1
    rows, heads = [], []
    if isinstance(chain, fi.ProteinChain):
      for a3m in (chain.paired_msa, chain.unpaired_msa):
        seqs, descs = parsers.parse_fasta(a3m or '')
        rows += [r for r in seqs if r]
        heads += [d for r, d in zip(seqs, descs) if r]
    if len(rows) > 1:
      # chai's per-source quota (parse_aligned_pqt_to_msa_context): at most 9999 UNIREF90 rows, BFD_UNICLUST unlimited
      uniref = np.cumsum([h.startswith('UniRef') for h in heads[1:]])
      kept = [0] + [i + 1 for i, h in enumerate(heads[1:]) if not h.startswith('UniRef') or uniref[i] <= _CHAI_UNIREF_QUOTA]
      rows, heads = [rows[i] for i in kept], [heads[i] for i in kept]
      cols = sel[np.sort(np.unique(resi[sel], return_index=True)[1])]   # first token of each residue
      want = {}
      for r, h in zip(rows[1:], heads[1:]):
        key = bytes(to_cls.get(c, vocab.index(residue_names.UNK)) for c in r if not c.islower())
        want.setdefault(key, 2 if h.startswith('UniRef') else 0)
      n_paired = int(np.asarray(batch.get('num_paired_alignments', 0)))
      for k in np.flatnonzero(mask[:, cols].any(-1))[1:]:
        hit = want.get(bytes(msa[k, cols].astype(np.uint8)))
        if hit is not None and (msa[k, cols] != to_cls['-']).any():
          source[k] = hit
        elif hit is None and k >= n_paired and len(kept) < len(uniref) + 1:
          mask[k, cols] = False                     # a UniRef row past the quota
    if len(rows) > 1:
      tok, dl = _chai_tokenise([chain.sequence] + rows)
      _, keep = np.unique(tok, axis=0, return_index=True)
      keep = np.sort(keep)
      tok, dl = tok[keep], dl[keep]
      counts = np.stack([(tok == c).sum(0) for c in range(profile.shape[-1])], -1) % 256
      profile[sel] = (counts / np.maximum(counts.sum(-1, keepdims=True), 1))[col]
      del_mean[sel] = dl.mean(0)[col]
      continue
    mods = dict((pos, ccd) for ccd, pos in
                (chain.ptms if isinstance(chain, fi.ProteinChain) else chain.modifications))
    text = ''.join(f'({mods[i + 1]})' if i + 1 in mods else c for i, c in enumerate(chain.sequence))
    query = _chai_tokenise([text])[0][0]
    q = query[np.minimum(col, query.size - 1)]
    msa[0, sel], msa[1:, sel], mask[1:, sel], dele[:, sel] = q, 31, False, 0
    profile[sel] = np.eye(profile.shape[-1], dtype=np.float32)[q]
    del_mean[sel] = 0
  batch['profile'] = profile.astype(np.asarray(batch['profile']).dtype)
  batch['deletion_mean'] = del_mean.astype(np.asarray(batch['deletion_mean']).dtype)
  # chai drops token-identical rows of the joined MSA (AF3 keeps the paired and the unpaired query, and duplicates)
  live = np.flatnonzero(mask.any(-1))
  _, first = np.unique(msa[live], axis=0, return_index=True)
  mask[np.setdiff1d(live, live[first])] = False
  batch['msa'], batch['msa_mask'], batch['deletion_matrix'] = msa, mask, dele
  batch['msa_source'] = source
  return batch


def _empty_template_gap(batch, gap_index=21, slots='first'):
  """A vendor's "no template" slot is a GAP template, not a zero template.

  Both pipelines pad the template axis to 4 slots and mask every atom, so the
  distogram, the unit vector and both masks come out zero either way. The
  RESTYPE one-hot does not: AF3's `empty_template_features` fills `aatype` with
  0, protenix's featuriser fills its one empty template with the GAP class and
  zero-pads the other three. Measured on native protenix1 at ubiquitin, its
  template_aatype is exactly that -- slot 0 all 31 (gap), slots 1-3 all 0.

  So the term is applied only when there is no real template anywhere, which is
  the case these vendors build that way; a batch carrying a real template is
  left alone.

  `slots` is the vendor's own padding, and the two differ:

    'first'  protenix -- ONE empty template with the gap restype, then three
             zero-padded slots ([31, 0, 0, 0] measured off its dump).
    'all'    opendde -- `make_dummy_feature` fills the WHOLE (4, N) block with
             31, commented "# gap" (data/utils.py), so every slot is a gap
             template.
    'atoms'  openfold3/openbind0 -- `create_template_feature_precursor_of3` starts every (slot, token) as GAP and
             writes only what a template covers, so an empty slot, and a chain without templates in a complex, read
             GAP even when another chain or slot has a real template; AF3 pads those with 0.

  `gap_index` is in OUR vocabulary: `template_modules._AF3_TO_OF3` maps 21 -> 31,
  which is the class the converted `a_proj` column expects.
  """
  aatype = batch.get('template_aatype')
  mask = batch.get('template_atom_mask')
  if aatype is None or mask is None:
    return
  aatype, mask = np.asarray(aatype), np.asarray(mask)
  if slots == 'atoms':
    if aatype.shape[0]:
      batch['template_aatype'] = np.where(mask.any(-1), aatype, gap_index).astype(aatype.dtype)
    return
  if not aatype.shape[0] or mask.any():
    return                                  # no slots, or a real template
  aatype = aatype.copy()
  if slots == 'all':
    aatype[:] = gap_index
  else:
    aatype[0] = gap_index
  batch['template_aatype'] = aatype


def _dna_gap_msa(batch):
  """OpenFold3 adds query rows for protein and RNA only (colabfold_msa_server): a DNA chain reads gap in every MSA row
  and a zero profile and deletion mean; a ligand token's profile is zero too (ours: a gap one-hot)."""
  from alphafold3.constants import residue_names
  dna = np.asarray(batch['is_dna']).astype(bool).reshape(-1)
  prof = np.array(batch['profile'])
  prof[dna | np.asarray(batch['is_ligand']).astype(bool).reshape(-1)] = 0.0
  batch['profile'] = prof
  if not dna.any():
    return batch
  msa = np.array(batch['msa'])
  msa[:, dna] = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP).index(residue_names.GAP)
  batch['msa'] = msa
  for k in ('deletion_matrix', 'deletion_mean'):
    v = np.array(batch[k])
    v[..., dna] = 0
    batch[k] = v
  if not any(np.asarray(batch[k]).any() for k in ('is_protein', 'is_rna') if k in batch):
    _dedupe_self_msa(batch)           # no protein or RNA: OF3's MSA is the one dummy row
  return batch


def _a3m_row_parser():
  """parse(line): an a3m row's class per aligned column and the insertions (lower case) before each."""
  import string
  from alphafold3.constants import residue_names
  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  lut = np.full(256, vocab.index(residue_names.UNK), np.int32)
  for c in string.ascii_uppercase:
    lut[ord(c)] = vocab.index(residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(c, residue_names.UNK))
  lut[ord('-')] = vocab.index(residue_names.GAP)

  def parse(line):
    a = np.frombuffer(line.encode(), np.uint8)
    ins = (a >= ord('a')) & (a <= ord('z'))
    keep = np.flatnonzero(~ins)
    cum = np.concatenate([[0], np.cumsum(ins)])[keep]
    return lut[a[keep]], np.diff(np.concatenate([[0], cum]))
  return parse


def _set_msa(batch, msa, dele, real, gap):
  """Writes a rebuilt MSA, every row live on the real tokens, over the batch's, padded with masked gap rows to the depth
  the batch had; returns that padding for the caller's other per-row features."""
  n_row = msa.shape[0]
  rows = max(n_row, np.asarray(batch['msa']).shape[0])
  pad = lambda x, v: np.concatenate([x, np.full((rows - n_row,) + x.shape[1:], v, x.dtype)]) if rows > n_row else x
  dt = np.asarray(batch['deletion_matrix']).dtype
  lim = np.iinfo(dt).max if np.issubdtype(dt, np.integer) else np.inf
  batch['msa'] = pad(msa, gap).astype(np.asarray(batch['msa']).dtype)
  batch['msa_mask'] = pad(np.broadcast_to(real, msa.shape), False).astype(np.asarray(batch['msa_mask']).dtype)
  batch['deletion_matrix'] = pad(np.minimum(dele, lim), 0).astype(dt)     # int8 as AF3's safe_cast_int8
  if 'num_alignments' in batch:
    batch['num_alignments'] = np.asarray(n_row, np.asarray(batch['num_alignments']).dtype)
  return pad


_BOLTZ_MAX_MSA_SEQS = 8192    # `boltz predict --max_msa_seqs`: parse_csv keeps this many unique rows per chain
_BOLTZ_MAX_PAIRS, _BOLTZ_MAX_TOTAL = 8192, 16384    # construct_paired_msa; inference max_seqs is const.max_msa_seqs


def _boltz_msa(batch, fold_input):
  """Boltz-2's MSA as `boltz predict` builds it from a chain's key,sequence CSV (query, paired rows keyed 1.., then
  the unpaired rows): parse_csv drops repeats of a sequence (gaps removed, insertions upper-cased) and stops at 8192
  rows; construct_paired_msa lays rows sharing a key across chains side by side (msa_paired 1 for those chains), fills
  the other chains of a paired row and then the unpaired rows from each chain's remaining sequences, to 16384. Its
  `visited` set never matches, so a paired sequence comes round again in the unpaired block, as in AF3. A chain
  without an MSA (ligand, nucleic, protein without alignments) has its query (UNK on a ligand) and then gaps. Profile
  and deletion mean are over every row, the latter of the deletion value (pi / 2) * arctan(d / 3)."""
  import collections
  from alphafold3.common import folding_input as fi
  from alphafold3.constants import residue_names
  from alphafold3.data import parsers

  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  gap, unk = vocab.index(residue_names.GAP), vocab.index(residue_names.UNK)
  parse = _a3m_row_parser()
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  real = np.asarray(batch['seq_mask']).astype(bool).reshape(-1)
  lig = np.asarray(batch['is_ligand']).astype(bool).reshape(-1)
  query = np.asarray(batch['msa'])[0].copy()
  query[lig & real] = unk
  chains = [i for i in range(len(fold_input.chains)) if (real & (asym == i + 1)).any()]
  seqs = {}
  for i in chains:
    c = fold_input.chains[i]
    if not isinstance(c, fi.ProteinChain):
      continue
    paired = parsers.parse_fasta(c.paired_msa or '')[0][1:]
    unpaired = parsers.parse_fasta(c.unpaired_msa or '')[0][1:]
    if not paired and not unpaired:
      continue
    seen, rows = set(), []
    # the CSV's query row is the chain sequence as typed (a modified position keeps its letter)
    typed = getattr(c, '_sequence', c.sequence)
    for key, line in [(None, typed)] + [(k + 1, r) for k, r in enumerate(paired)] + [(None, r) for r in unpaired]:
      line = line.strip()
      flat = line.replace('-', '').upper()
      if not line or flat in seen:
        continue
      seen.add(flat)
      rows.append((key, line))
      if len(rows) >= _BOLTZ_MAX_MSA_SEQS:
        break
    seqs[i] = rows
  # construct_paired_msa
  taxonomy = {}
  for i in chains:
    for j, (key, _) in enumerate(seqs.get(i, ())):
      if key is not None:
        taxonomy.setdefault(key, []).append((i, j))
  taxonomy = sorted([v for v in taxonomy.values() if len(v) > 1], key=lambda v: len({i for i, _ in v}), reverse=True)
  available = {i: collections.deque(range(1, len(seqs.get(i, [None])))) for i in chains}
  pairing, is_paired = [{i: 0 for i in chains}], [{i: 1 for i in chains}]
  for pairs in taxonomy:
    occ = {}
    for i, j in pairs:
      occ.setdefault(i, []).append(j)
    for k in range(max(len(v) for v in occ.values())):
      row = {i: v[k % len(v)] for i, v in occ.items()}
      flag = {i: int(i in row) for i in chains}
      for i in chains:
        if i not in row:
          row[i] = available[i].popleft() if available[i] else -1
      pairing.append(row)
      is_paired.append(flag)
      if len(pairing) >= _BOLTZ_MAX_PAIRS:
        break
    if len(pairing) >= _BOLTZ_MAX_PAIRS:
      break
  left = max(len(v) for v in available.values())
  for _ in range(min(_BOLTZ_MAX_TOTAL - len(pairing), left)):
    pairing.append({i: available[i].popleft() if available[i] else -1 for i in chains})
    is_paired.append({i: 0 for i in chains})
  pairing, is_paired = pairing[:_BOLTZ_MAX_TOTAL], is_paired[:_BOLTZ_MAX_TOTAL]
  n_row, n_tok = len(pairing), asym.size
  msa = np.full((n_row, n_tok), gap, np.int32)
  dele = np.zeros((n_row, n_tok), np.float32)
  flag = np.zeros((n_row, n_tok), np.float32)
  msa[0] = query
  for i in chains:
    sel = np.flatnonzero(real & (asym == i + 1))
    flag[:, sel] = np.asarray([p[i] for p in is_paired], np.float32)[:, None]
    if i not in seqs:
      continue
    col = resi[sel] - 1
    idx = np.asarray([p[i] for p in pairing])
    for r in np.flatnonzero(idx > 0):
      tok, d = parse(seqs[i][idx[r]][1])
      if tok.size > col.max():          # boltz indexes by residue; a row shorter than its chain is malformed, skip it
        msa[r, sel], dele[r, sel] = tok[col], d[col]
  n_cls = np.asarray(batch['profile']).shape[-1]
  counts = np.bincount((np.arange(n_tok) * n_cls + np.minimum(msa, n_cls - 1)).ravel(),
                       minlength=n_tok * n_cls).reshape(n_tok, n_cls).astype(np.float32)
  pad = _set_msa(batch, msa, dele, real, gap)
  batch['msa_paired'] = pad(flag, 0)
  batch['profile'] = (counts / n_row).astype(np.asarray(batch['profile']).dtype)
  batch['deletion_mean'] = ((np.pi / 2) * np.arctan(dele / 3.0)).mean(0).astype(np.asarray(batch['deletion_mean']).dtype)
  return batch


_RF3_MAX_MSA_SEQS = 10000      # rf3 build_af3_transform_pipeline max_msa_sequences and n_msa


def _rf3_msa(batch, fold_input):
  """RoseTTAFold3's MSA as atomworks builds it from one a3m per protein entity (query, paired rows, unpaired rows):
  parse_a3m keeps the first 10000 sequences, no deduplication. One entity with an MSA is used as read. Several are
  joined by TaxID (ColabFold headers carry none, so only the query row pairs) and, dense, each entity's other rows
  lose those that are all gap in its own columns -- the paired block's empty rows, which ends ColabFold's row
  alignment -- and are stacked side by side, padded with gaps. A chain without an MSA has its query (restype) and then
  padding. Profile and insertion mean count non-padded rows only; the mean is of raw insertion counts."""
  from alphafold3.common import folding_input as fi
  from alphafold3.constants import residue_names
  from alphafold3.data import parsers

  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  gap, unk = vocab.index(residue_names.GAP), vocab.index(residue_names.UNK)
  parse = _a3m_row_parser()
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  real = np.asarray(batch['seq_mask']).astype(bool).reshape(-1)
  query = np.asarray(batch['msa'])[0].copy()
  aatype = np.asarray(batch['aatype']).reshape(-1)
  # entities with an MSA, in chain order: (sequence, paired, unpaired) -> rows
  entity_rows, chain_entity = {}, {}
  for i, c in enumerate(fold_input.chains):
    if not isinstance(c, fi.ProteinChain):
      continue
    paired = parsers.parse_fasta(c.paired_msa or '')[0][1:]
    unpaired = parsers.parse_fasta(c.unpaired_msa or '')[0][1:]
    if not paired and not unpaired:
      continue
    key = (c.sequence, c.paired_msa, c.unpaired_msa)
    if key not in entity_rows:
      lines = [r.strip() for r in paired + unpaired if r.strip()][:_RF3_MAX_MSA_SEQS - 1]
      entity_rows[key] = [parse(r) for r in lines]
    chain_entity[i] = key
  if len(entity_rows) > 1:
    entity_rows = {k: [(t, d) for t, d in v if (t != gap).any()] for k, v in entity_rows.items()}
  n_row = 1 + max([len(v) for v in entity_rows.values()], default=0)
  n_tok = asym.size
  msa = np.full((n_row, n_tok), gap, np.int32)
  dele = np.zeros((n_row, n_tok), np.float32)
  padded = np.ones((n_row, n_tok), bool)
  padded[0] = False
  msa[0] = np.where(query == gap, aatype, query)     # a token with no alignment reads its own restype in row 0
  # an atomised residue (several tokens on one residue) is a ligand to atomworks: no alignment, and UNK in row 0
  # unless it is a standard residue (atomised for a bond to a ligand), which keeps its own letter
  _, inv, cnt = np.unique(np.stack([asym, resi]), axis=1, return_inverse=True, return_counts=True)
  atomised = real & (cnt[inv.reshape(-1)] > 1)
  layout = batch.get('token_atoms_layout')
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  if layout is not None and atomised.any():
    from alphafold3.model.pipeline import chiral_features
    standard = chiral_features._standard_residues()
    amask = np.asarray(batch['ref_mask']).astype(bool)
    for t in np.flatnonzero(atomised):
      a = np.flatnonzero(amask[t])
      if not a.size or str(layout.res_name[t, a[0]]) not in standard:
        msa[0, t] = unk
  for i, key in chain_entity.items():
    sel = np.flatnonzero(real & (asym == i + 1) & ~atomised)
    col = resi[sel] - 1
    for r, (tok, d) in enumerate(entity_rows[key], start=1):
      if tok.size > col.max():
        msa[r, sel], dele[r, sel] = tok[col], d[col]
      padded[r, sel] = False
  live = ~padded
  n_cls = np.asarray(batch['profile']).shape[-1]
  counts = np.bincount((np.arange(n_tok) * n_cls + np.minimum(msa, n_cls - 1))[live],
                       minlength=n_tok * n_cls).reshape(n_tok, n_cls).astype(np.float32)
  n_live = np.maximum(live.sum(0), 1)
  _set_msa(batch, msa, dele, real, gap)
  batch['profile'] = (counts / n_live[:, None]).astype(np.asarray(batch['profile']).dtype)
  batch['deletion_mean'] = ((dele * live).sum(0) / n_live).astype(np.asarray(batch['deletion_mean']).dtype)
  return batch


def _ccd_restype(batch, fold_input):
  """Protenix's restype for a non-standard residue (add_cano_seq_resname): the CCD one_letter_code read for the
  residue's own CCD type (peptide -> the amino acid or UNK, DNA -> D + letter or DN, RNA -> letter or N, else UNK), where
  AF3 takes the CCD parent -- PCA is GLN there (one letter Q), GLU here (parent GLU)."""
  from alphafold3.constants import chemical_components, residue_names
  layout = batch.get('token_atoms_layout')
  if layout is None or 'aatype' not in batch:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  ccd = chemical_components.Ccd(user_ccd=getattr(fold_input, 'user_ccd', None) if fold_input is not None else None)
  standard = _standard_residue_names()
  vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  amask = np.asarray(batch['ref_mask']).astype(bool)
  aatype = np.array(batch['aatype'], copy=True)
  cache = {}
  for t in range(aatype.shape[0]):
    a = np.flatnonzero(amask[t])
    if not a.size:
      continue
    name = str(layout.res_name[t, a[0]])
    if name in standard:
      continue
    if name not in cache:
      entry = ccd.get(name) or {}
      kind = ''.join(entry.get('_chem_comp.type', ())).upper()
      one = ''.join(entry.get('_chem_comp.one_letter_code', ())).strip()
      one = one if len(one) == 1 and one != '?' else ''
      if 'PEPTIDE' in kind and kind != 'PEPTIDE-LIKE':
        res = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(one, residue_names.UNK)
      elif 'DNA' in kind:
        res = 'D' + one if 'D' + one in vocab else residue_names.UNK_DNA
      elif 'RNA' in kind:
        res = one if one in ('A', 'G', 'C', 'U') else residue_names.UNK_RNA
      else:
        res = residue_names.UNK
      cache[name] = vocab.index(res) if res in vocab else vocab.index(residue_names.UNK)
    aatype[t] = cache[name]
  batch['aatype'] = aatype.astype(np.asarray(batch['aatype']).dtype)
  return batch


def _dedupe_self_msa(batch, query_once=False):
  """ESMFold2's self-MSA is the query ONCE, not the query twice.

  AF3 concatenates a paired and an unpaired MSA, and each one begins with the
  query -- so a chain with no alignments at all arrives with `num_alignments 2`
  and two IDENTICAL unmasked rows. Native ESMFold2's own featuriser emits a
  depth-1 MSA for that input: its dump carries `feat.msa` at (1, 1, 68), and
  `esmfold2_reference.self_msa` is the query alone.

  Two identical rows are NOT free, even though an outer-product MEAN over
  duplicates is unchanged. Measured on 6MRR at one pass: the trunk's MSA
  injection reads std 619.2 against native's 593.9 (corr 0.998823), and feeding
  the reference the same duplicated row reproduces our number to 3.7e-07 --
  which is what identified the depth, rather than the encoder, as the
  difference. The pair-weighted averaging and the row transition are what make
  it depth-sensitive.

  IT FIRES ONLY ON A SELF-MSA -- one where EVERY live row is the query. With a
  real alignment the duplicate is NOT ours to drop: protenix builds a paired and
  an unpaired MSA exactly as AF3 does, so its own batch for a 4-row a3m carries
  FIVE rows (paired query, unpaired query, three homologs). The depth-1
  behaviour these vendors show is the NO-ALIGNMENT case, where their pipelines
  skip the MSA path and synthesise a single dummy row.

  Getting that wrong is expensive and quiet: dropping the duplicate alongside a
  real MSA cost 2.3 A on ubiquitin (ours 4.024 against native's 1.704, both
  stable across processes), and the earlier version of this docstring asserted
  the opposite -- "a real alignment whose first row happens to be the query is
  left alone" -- while the code dropped it anyway.
  """
  if 'msa' not in batch:
    return batch                      # a partial batch, as the unit tests build
  msa = np.asarray(batch['msa'])
  mask = np.asarray(batch['msa_mask'])
  live = np.flatnonzero(mask.any(-1))
  if live.size < 2:
    return batch
  a, b = live[0], live[1]
  if not np.array_equal(msa[a], msa[b]):
    return batch
  # query_once: ESMFold2 builds its MSA from the a3m alone, so beside a real alignment too the query appears once
  if not query_once and not all(np.array_equal(msa[a], msa[i]) for i in live[1:]):
    return batch                      # a real alignment: not a self-MSA
  m = mask.copy()
  m[b] = False
  batch['msa_mask'] = m
  if 'num_alignments' in batch:
    batch['num_alignments'] = np.asarray(
        np.asarray(batch['num_alignments']) - 1, dtype=np.asarray(
            batch['num_alignments']).dtype)
  return batch


def _esm_deletion_mean(batch):
  """ESMFold2's deletion_mean is the mean over MSA rows of its deletion VALUE, (pi / 2) * arctan(d / 3), not of the counts."""
  if 'deletion_matrix' not in batch or 'deletion_mean' not in batch:
    return batch
  live = np.asarray(batch['msa_mask']).any(-1)
  if not live.any():
    return batch
  value = (np.pi / 2) * np.arctan(np.asarray(batch['deletion_matrix'], np.float32)[live] / 3.0)
  batch['deletion_mean'] = value.mean(0).astype(np.asarray(batch['deletion_mean']).dtype)
  return batch


def _complex_msa_profile(batch, unk=20):
  """ESMFold2's profile is a mean over the whole complex MSA, where a chain without an MSA has its query (UNK on a ligand) in row 0 and gaps below."""
  if 'msa' not in batch:
    return batch
  from alphafold3.cpp import msa_profile
  msa = np.asarray(batch['msa']).copy()
  live = np.flatnonzero(np.asarray(batch['msa_mask']).any(-1))
  if live.size == 0:
    return batch
  lig = np.flatnonzero(np.asarray(batch['is_ligand']).astype(bool) & np.asarray(batch['seq_mask']).astype(bool))
  msa[live[0], lig] = unk
  batch['msa'] = msa.astype(np.asarray(batch['msa']).dtype)
  if 'profile' in batch:
    n = np.asarray(batch['profile']).shape[-1]
    batch['profile'] = msa_profile.compute_msa_profile(msa=np.minimum(msa[live], n - 1), num_residue_types=n).astype(np.float32)
  return batch


def _native_nonprotein_msa(batch, unk=20, gap=21):
  """OpenDDE's MSA on non-protein tokens: a ligand reads UNK in every row and profile, a nucleic chain's all-gap rows repeat its query."""
  if 'msa' not in batch:
    return
  msa = np.asarray(batch['msa']).copy()
  mask = np.asarray(batch['msa_mask']).copy()
  live = np.flatnonzero(mask.any(-1))
  lig = np.asarray(batch['is_ligand']).astype(bool)
  nuc = (np.asarray(batch['is_rna']) | np.asarray(batch['is_dna'])).astype(bool)
  real = np.asarray(batch['seq_mask']).astype(bool)
  asym = np.asarray(batch['asym_id'])
  cols = np.flatnonzero(lig & real)
  msa[np.ix_(live, cols)] = unk
  mask[np.ix_(live, cols)] = True
  for a in np.unique(asym[nuc & real]):
    c = np.flatnonzero((asym == a) & nuc & real)
    rows = live[(msa[np.ix_(live, c)] == gap).all(-1)]
    msa[np.ix_(rows, c)] = msa[live[0], c]
    mask[np.ix_(rows, c)] = True
  if 'profile' in batch:
    profile = np.asarray(batch['profile']).copy()
    profile[cols] = 0.0
    profile[cols, unk] = 1.0
    batch['profile'] = profile
  batch['msa'] = msa.astype(np.asarray(batch['msa']).dtype)
  batch['msa_mask'] = mask
  return batch


def _attach_esm(batch, esm):
  """Put ESM2 embeddings on the protein tokens, in order; everything else zero.

  chai-1's TOKEN feature stream is mostly ESM2 -- zeroing it drops the token
  embedding to corr 0.327 of chai's, and a natural protein folds to 5.70 A where
  chai reaches 0.642. `esm` is (num_protein_tokens, 2560) in token order, chai's
  own traced ESM whose raw last hidden state IS the feature verbatim.
  """
  is_protein = np.asarray(batch['is_protein']).astype(bool)
  out = np.zeros(is_protein.shape + (np.asarray(esm).shape[-1],), np.float32)
  idxs = np.flatnonzero(is_protein)
  rows = np.asarray(esm, np.float32)
  # ONE ROW PER RESIDUE, NOT PER TOKEN -- the same fix `_attach_lm_pair` needed,
  # and found the same way. A language model reads a SEQUENCE, so its rows are
  # residues; AF3 ATOMISES a modified residue into one token per atom, all of
  # them protein. Keying on the protein-token count raised `esm has 76 rows but
  # the batch has 85 protein tokens` on phospho-ubiquitin, so chai-1 could not
  # be given its ESM2 embeddings on ANY PTM input -- and chai without them folds
  # a different model (5.70 A where it reaches 0.642).
  #
  # Map each protein token to the row of ITS RESIDUE, which is AF3's own
  # parent-residue convention for an atomised residue's restype. With nothing
  # atomised every residue is one token and this is exactly the identity it
  # replaces.
  asym = np.asarray(batch['asym_id']).reshape(-1)[idxs]
  resi = np.asarray(batch['residue_index']).reshape(-1)[idxs]
  rank, order = {}, []
  for key in zip(asym.tolist(), resi.tolist()):
    order.append(rank.setdefault(key, len(rank)))
  if len(rows) == len(idxs):
    out[idxs] = rows
  elif len(rows) == len(rank):
    out[idxs] = rows[np.asarray(order, np.int64)]
  else:
    raise ValueError(f'esm has {len(rows)} rows but the batch has {len(idxs)} '
                     f'protein tokens spanning {len(rank)} residues')
  batch['esm_embeddings'] = out
  return batch


def _wide_key_window(batch, k_size, prefix=''):
  """Rebuild the atom-attention key window at a WIDER size than AF3's 128.

  AF3 gives each block of 32 queries 128 keys, which is a +/-48 window. ESMFold2
  attends +/-64, so 128 keys cannot hold its window however they are centred:
  the graph masks to +/-64 by rank, but a query can only see keys the SUBSET
  contains, and the subset was ten short of the window for a typical interior
  query (129 keys wanted, 119 median seen). Silent -- attention over a truncated
  key set is still well-formed attention.

  Rewritten on the finished batch, like _circular_key_window: the key gathers
  are index arrays over the flat queries layout, so the window is the only thing
  that changes.
  """
  return _key_window(batch, 'slide', k_size=k_size, prefix=prefix)

def _attach_lm_pair(batch, lm_pair):
  """ESMFold2's language-model PAIR representation, from alphafold3.model.esm.

  (num_tokens, num_tokens, c_z), built outside the fold from ESM-C's hidden
  states. It has to be in the SAME token order the featuriser laid out, which is
  why it is keyed to the batch rather than to the input JSON.

  A pair rep covering only the protein tokens is scattered into the full token grid with zeros elsewhere. Native
  ESMFold2 instead runs its shim over every token, zero hidden states off protein; `_attach_lm_hidden` does that.
  """
  lm = np.asarray(lm_pair, np.float32)
  n = np.asarray(batch['token_index']).shape[-1]
  if lm.shape[:2] != (n, n):
    is_protein = np.asarray(batch['is_protein']).astype(bool)
    idxs = np.flatnonzero(is_protein)
    # ONE ROW PER RESIDUE, NOT PER TOKEN. The language model reads a sequence,
    # so its rows are residues; AF3 ATOMISES a modified residue into one token
    # per atom, all of them protein. Keying on the protein-token COUNT therefore
    # failed on any PTM -- `lm_pair is (76, 76) but the batch has 85 tokens
    # (85 of them protein)` on phospho-ubiquitin -- and ESMFold2 could not be
    # given its language model at all on such an input.
    #
    # So map each protein token to the row of ITS RESIDUE, which is the same
    # parent-residue convention AF3 uses for an atomised residue's restype. With
    # nothing atomised every residue is one token and this is exactly the
    # identity it replaces.
    asym = np.asarray(batch['asym_id']).reshape(-1)[idxs]
    resi = np.asarray(batch['residue_index']).reshape(-1)[idxs]
    rank, rows = {}, []
    for key in zip(asym.tolist(), resi.tolist()):
      rows.append(rank.setdefault(key, len(rank)))
    rows = np.asarray(rows, np.int64)
    if lm.shape[:2] != (len(rank), len(rank)):
      raise ValueError(
          f'lm_pair is {lm.shape[:2]} but the batch has {n} tokens '
          f'({len(idxs)} of them protein, spanning {len(rank)} residues)')
    full = np.zeros((n, n, lm.shape[-1]), np.float32)
    full[np.ix_(idxs, idxs)] = lm[np.ix_(rows, rows)]
    lm = full
  batch['lm_pair'] = lm
  return batch


def _attach_lm_hidden(batch, rows, shim_params):
  """ESMFold2's language-model pair as native builds it: ESM-C's states (residues, layers, width) on every token,
  zeros where a token is not protein (and with no protein chain at all), then the shim over the whole token grid.
  A protein token takes its residue's row, which covers an atomised modified residue.
  """
  from alphafold3.model import esm

  n = np.asarray(batch['token_index']).shape[-1]
  layers, width = np.shape(shim_params['combine'])[0], np.shape(shim_params['lm_norm/scale'])[0]
  hidden = np.zeros((n, layers, width), np.float32)
  idxs = np.flatnonzero(np.asarray(batch['is_protein']).astype(bool))
  if rows is not None and len(idxs):
    asym = np.asarray(batch['asym_id']).reshape(-1)[idxs]
    resi = np.asarray(batch['residue_index']).reshape(-1)[idxs]
    rank, order = {}, []
    for key in zip(asym.tolist(), resi.tolist()):
      order.append(rank.setdefault(key, len(rank)))
    if len(rows) != len(rank):
      raise ValueError(f'ESM-C has {len(rows)} rows but the batch has {len(idxs)} protein tokens spanning '
                       f'{len(rank)} residues')
    hidden[idxs] = np.asarray(rows, np.float32)[np.asarray(order, np.int64)]
  batch['lm_pair'] = np.asarray(esm.shim(hidden, shim_params), np.float32)
  return batch


def _centre_conformers(batch):
  """Subtract each conformer group's own mean from ref_pos -- see
  model_config.CENTRE_REF_CONFORMERS for who does this and why it matters.

  Grouped by `ref_space_uid`, which is the grouping those natives use, and
  MASKED: an unmasked mean would drag the centre toward padding atoms. Padding
  keeps zero afterwards, because the mask is what every consumer reads and a
  nonzero coordinate there would put real numbers where the graph expects none.
  """
  if 'ref_pos' not in batch:
    # Nothing to centre. A real featurised batch always carries the reference
    # structure, but a caller can hand `apply` a partial batch -- the unit
    # tests build atom-window-only batches -- and a featurisation step should
    # skip a convention it has no input for rather than raise.
    return
  pos = np.array(batch['ref_pos'], dtype=np.float32)
  mask = np.asarray(batch['ref_mask']) > 0
  uid = np.asarray(batch['ref_space_uid'])
  flat = pos.reshape(-1, 3)
  fm = mask.reshape(-1)
  fu = uid.reshape(-1).astype(np.int64)
  if not fu.size:
    return
  n = int(fu.max()) + 1
  w = fm.astype(np.float32)
  count = np.bincount(fu, weights=w, minlength=n)
  centre = np.stack(
      [np.bincount(fu, weights=w * flat[:, k], minlength=n) for k in range(3)],
      axis=-1) / np.maximum(count, 1.0)[:, None]
  flat = (flat - centre[fu]) * w[:, None]
  batch['ref_pos'] = flat.reshape(pos.shape)


def _translate_conformers(batch, max_translation=1.0):
  """Protenix's inference ref_pos augmentation after the centring: each conformer group moves by U(-1, 1) per axis."""
  if 'ref_pos' not in batch:
    return
  pos = np.array(batch['ref_pos'], dtype=np.float32)
  mask = np.asarray(batch['ref_mask']) > 0
  uid = np.asarray(batch['ref_space_uid']).reshape(-1).astype(np.int64)
  if not uid.size:
    return
  # seeded by the per-seed conformers, so every seed is reproducible; native draws from numpy's global state
  import hashlib
  rng = np.random.default_rng(np.frombuffer(hashlib.sha256(pos.tobytes()).digest()[:8], np.uint64)[0])
  shift = rng.uniform(-max_translation, max_translation, size=(int(uid.max()) + 1, 3)).astype(np.float32)
  flat = pos.reshape(-1, 3) + shift[uid] * mask.reshape(-1)[:, None]
  batch['ref_pos'] = flat.reshape(pos.shape)


# threads per SM by compute capability, for torch's launch grid; the SM count comes from the device
_THREADS_PER_SM = {(7, 0): 2048, (7, 5): 1024, (8, 0): 2048, (8, 6): 1536, (8, 7): 1536, (8, 9): 1536,
                   (9, 0): 2048, (10, 0): 2048, (10, 3): 2048, (12, 0): 1536, (12, 1): 1536}
NATIVE_RNG_PASSES = 64


def _device_grid_params():
  """(SM count, max threads per SM) of the device this process runs on, for torch's dropout/randperm grids"""
  import jax
  dev = jax.devices()[0]
  sms = int(getattr(dev, 'core_count', 0) or 1)
  try:
    cc = tuple(int(x) for x in str(getattr(dev, 'compute_capability', '8.0')).split('.')[:2])
  except ValueError:
    # a ROCm device reports its gfx name: no CUDA launch grid to mirror, protenix folded nowhere on AMD
    cc = None
  return sms, _THREADS_PER_SM.get(cc, 2048)


def _msa_depth(batch):
  """rows of the MSA with any real position"""
  n_tok = int(np.asarray(batch['seq_mask']).sum())
  return int(np.asarray(batch['msa_mask']).astype(bool)[:, :n_tok].any(-1).sum())


def _crop_msa_padding(batch):
  """For a model that runs every MSA row it is given: crop the padding to a power-of-two bucket over the real rows"""
  if 'msa' not in batch:
    return
  mask = np.asarray(batch['msa_mask'])
  rows = mask.shape[0]
  depth = min(rows, max(1024, 1 << max(0, _msa_depth(batch) - 1).bit_length()))
  # live rows first, in order: chai's dedupe masks duplicates in place, and a plain [:depth] cut real rows past them
  # (msa_gappy kept 4139 of its 6529)
  live = mask.any(-1)
  order = np.concatenate([np.flatnonzero(live), np.flatnonzero(~live)])[:depth]
  for key in ('msa', 'msa_mask', 'deletion_matrix', 'msa_source'):
    if key in batch:
      batch[key] = np.asarray(batch[key])[order]


def _native_rng(batch, seed, pair_channel, fold_input):
  """Protenix's MC dropout and per-pass MSA subsets as native draws them for this seed -- see protenix_rng."""
  from alphafold3.common import folding_input
  from alphafold3.constants import chemical_components
  from alphafold3.model.pipeline import protenix_rng
  polymers = (folding_input.ProteinChain, folding_input.RnaChain, folding_input.DnaChain)
  # native builds each entity once, and copies of a chain are one entity
  entities = list(dict.fromkeys((type(c), tuple(c.to_ccd_sequence())) for c in fold_input.chains
                                if isinstance(c, polymers)))
  draws = protenix_rng.leaving_group_draws([codes for _, codes in entities],
                                           chemical_components.Ccd(user_ccd=fold_input.user_ccd))
  n_tok = int(np.asarray(batch['seq_mask']).sum())
  n_msa = _msa_depth(batch)
  sms, tps = _device_grid_params()
  mc, rows, offsets = protenix_rng.replay(
      seed, leaving_draws=draws, n_msa=n_msa, n_tokens=n_tok, pair_channel=pair_channel,
      passes=NATIVE_RNG_PASSES, sms=sms, threads_per_sm=tps)
  keep = np.zeros((NATIVE_RNG_PASSES, np.asarray(batch['msa_mask']).shape[0]), bool)
  for k, r in enumerate(rows):
    keep[k, r] = True
  batch['px_seed'] = np.asarray(seed, np.int32)
  batch['px_mc_dropout'] = np.asarray(mc, np.int32)
  batch['px_msa_rows'] = keep
  batch['px_drop_offset'] = np.asarray(offsets, np.int32)
  batch['px_n_tokens'] = np.asarray(n_tok, np.int32)
  batch['px_grid'] = np.asarray(protenix_rng.grid_size(n_tok * n_tok * pair_channel, sms, tps), np.int32)


def _ligand_bonded_residues(fold_input):
  """(chain_id, res_id) of every polymer residue a declared bond ties to a ligand."""
  polymers = {c.id for c in fold_input.chains if type(c).__name__ in ('ProteinChain', 'RnaChain', 'DnaChain')}
  out = set()
  for a, b in getattr(fold_input, 'bonded_atom_pairs', None) or ():
    if (a[0] in polymers) != (b[0] in polymers):
      poly = a if a[0] in polymers else b
      out.add((str(poly[0]), int(poly[1])))
  return frozenset(out)


def _largest_modified_residue(fold_input):
  """Slots for the biggest modified polymer residue (CCD heavy atoms), so a model that keeps one on a single token, as
  boltz does whatever its size, never truncates or splits it; None when 24 suffice."""
  from alphafold3.common import folding_input as fi
  from alphafold3.constants import chemical_components
  ccd = chemical_components.Ccd(user_ccd=getattr(fold_input, 'user_ccd', None))
  most = 0
  for chain in fold_input.chains:
    mods = (chain.ptms if isinstance(chain, fi.ProteinChain) else
            chain.modifications if isinstance(chain, (fi.RnaChain, fi.DnaChain)) else ())
    for name, _ in mods:
      entry = ccd.get(name) or {}
      most = max(most, sum(e not in ('H', 'D') for e in entry.get('_chem_comp_atom.type_symbol', ())))
  return -(-most // 8) * 8 if most > 24 else None


def prepare_input(fold_input, spec):
  """The input-side conventions, applied before featurisation: a copy of `fold_input` as the model's featuriser sees it.

  dedupe_unpaired_msa: the protenix featuriser deduplicates each protein chain's unpaired MSA, ignoring insertions,
  where AF3 keeps a supplied MSA as given (msa_featurizer RawMsa, dedup=True); the paired MSA stays as given in both.
  of3_main_msa: OpenFold3 reads the first N rows of a protein or RNA chain's main MSA (max_seq_counts colabfold_main) and
  deduplicates them, ignoring insertions, before its profile and deletion mean; AF3 keeps every row and profiles them all.
  ccd_unk: an unknown amino acid keeps the CCD UNK atoms (N CA C O CB CG) where AF3 builds it as GLY ('cb': N CA C O CB,
  boltz); process-wide, set on
  every call so the next featurisation follows this model.
  atomize_ligand_bonded: a polymer residue a declared bond ties to a ligand is one token per atom (atomworks'
  covalent_modifications); process-wide like ccd_unk.
  drop_inner_terminal_atoms: True -- OXT only at a chain's last residue and OP3 only at its first, also on modified
  residues (atomworks); 'leaving' -- a modified polymer residue's CCD leaving atoms unless it ends the chain (esm);
  'leaving_all' -- every non-standard polymer residue's CCD leaving atoms, ends included (of3);
  'chai' -- OXT/OXT' off every non-standard amino acid, OP3 kept on every non-standard nucleotide (chai);
  process-wide like ccd_unk, applied by the pipeline before tokenising.
  """
  knobs = spec.featurise
  from alphafold3.model import features as af3_features
  from alphafold3.structure import chemical_components as struc_chem_comps
  struc_chem_comps.UNK_AS_GLY = 'ALA' if knobs.get('ccd_unk') == 'cb' else not knobs.get('ccd_unk')
  af3_features.ATOMIZED_RESIDUES = (_ligand_bonded_residues(fold_input) if knobs.get('atomize_ligand_bonded')
                                    else frozenset())
  from alphafold3.model.pipeline import pipeline as af3_pipeline
  af3_pipeline.INNER_TERMINAL_ATOMS = knobs.get('drop_inner_terminal_atoms') or False
  af3_pipeline.MAX_ATOMS_PER_TOKEN = (_largest_modified_residue(fold_input) if knobs.get('modified_as_one_token')
                                      else None)
  cap = knobs.get('of3_main_msa')
  if not (knobs.get('dedupe_unpaired_msa') or cap or knobs.get('pair_as_unpair')):
    return fold_input
  import dataclasses
  import string
  from alphafold3.common import folding_input
  from alphafold3.data import parsers
  strip = str.maketrans('', '', string.ascii_lowercase)

  def dedupe(a3m):
    seqs, descs = parsers.parse_fasta(a3m)
    seen, lines = set(), []
    for s, d in list(zip(seqs, descs))[:cap]:
      if s.translate(strip) not in seen:
        seen.add(s.translate(strip))
        lines += [f'>{d}', s]
    return '\n'.join(lines) + '\n'

  chains = []
  for c in fold_input.chains:
    if isinstance(c, folding_input.ProteinChain) and (c.unpaired_msa or (knobs.get('pair_as_unpair') and c.paired_msa)):
      unpaired = c.unpaired_msa or ''
      if knobs.get('pair_as_unpair') and c.paired_msa:
        # protenix msa_pair_as_unpair: the paired rows join the unpaired MSA (paired first), deduplicated
        unpaired = c.paired_msa.rstrip('\n') + '\n' + unpaired
      # the sequence as typed (.sequence is the CCD-parent letters)
      c = folding_input.ProteinChain(id=c.id, sequence=getattr(c, '_sequence', c.sequence), ptms=c.ptms,
                                     description=c.description, unpaired_msa=dedupe(unpaired), paired_msa=c.paired_msa,
                                     templates=c.templates)
    elif cap and isinstance(c, folding_input.RnaChain) and c.unpaired_msa:
      c = folding_input.RnaChain(id=c.id, sequence=getattr(c, '_sequence', c.sequence), modifications=c.modifications,
                                 description=c.description, unpaired_msa=dedupe(c.unpaired_msa))
    chains.append(c)
  return dataclasses.replace(fold_input, chains=chains)


def apply(batch, spec, *, refeaturise=None, model_dir=None, esm=None,
          has_msa=True, fold_input=None, cyclic=None, lm_pair=None, seed=None, lm_hidden=None):
  """Apply `spec`'s featurisation conventions to a featurised batch, in place.

  spec is a model_registry.ModelSpec; the work is driven by spec.featurise.
  A model with no entries there (alphafold3, openfold3, intellifold2) passes
  through untouched.

  refeaturise is a zero-argument callable that repeats the featurisation that
  produced `batch`. Only opendde needs it: its structural-token layout is built
  by re-running the featuriser under a capture of the tokenizer's internals.

  model_dir is where this model's weights live; chai-1 reads its own standard
  residue conformers from there.

  cyclic is not a per-model convention at all -- it wraps the shared relative
  position encoding, so every family here honours it. It is applied whether or
  not the model declares anything.

  lm_hidden (ESM-C rows or None, shim params) builds ESMFold2's language-model pair over every token, as native
  does; lm_pair is a finished one.
  """
  knobs = spec.featurise
  if cyclic:
    cyclic_period(batch, cyclic, fold_input=fold_input)
  # A KNOB, not a name test, and the registry derives it from
  # model_config.CENTRE_REF_CONFORMERS so the two cannot drift. It has to be a
  # knob because every caller guards this whole step on `if spec.featurise:` --
  # a model with an empty entry never gets here at all, which silently skipped
  # openfold3 and openbind0 when this was a name test.
  import os
  if (knobs.get('centre_conformers')
      and not os.environ.get('AF3_NO_CENTRE_CONFORMERS')):
    _centre_conformers(batch)
  if knobs.get('translate_conformers') and not os.environ.get('AF3_NO_TRANSLATE_CONFORMERS'):
    _translate_conformers(batch)
  if not knobs:
    return batch

  if knobs.get('modified_as_one_token'):
    if fold_input is None:
      raise ValueError(f'{spec.name} needs fold_input= to find its modified '
                       'residues')
    # The single token is produced at featurisation time
    # (flatten_non_standard_residues=False); what is left is the pair of signals
    # boltz2 reads on it.
    _mark_modified_residues(batch, fold_input, unknown_restype=True)

  if knobs.get('atomize_ligand_bonded'):
    _ligand_bonded_units(batch, fold_input)
  if knobs.get('atomized_backbone_bonds'):
    _atomized_backbone_bonds(batch, fold_input, mode=knobs['atomized_backbone_bonds'])
  if knobs.get('declared_polymer_bonds'):
    _declared_polymer_bonds(batch, fold_input, rule=knobs['declared_polymer_bonds'])
  if knobs.get('declared_bond_feature'):
    _declared_bond_feature(batch, fold_input)
  if knobs.get('atomized_unknown_restype'):
    from alphafold3.model.pipeline import chiral_features
    # before restype_alignment, which reads `aatype` to rewrite profile and msa.
    mode = knobs['atomized_unknown_restype']
    chiral_features.apply_unknown_restype_on_atomized(
        batch, nucleic=mode in ('nucleic', 'esm'), nucleic_unknown=30 if mode == 'chai' else None,
        channels=('aatype',) if mode == 'aatype' else ('aatype', 'profile', 'msa'), msa_nucleic_only=mode == 'esm')
  if knobs.get('query_letter_msa'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.apply_query_letter_on_atomized(batch, fold_input)
  if knobs.get('ccd_restype'):
    _ccd_restype(batch, fold_input)
  if knobs.get('atomized_element_names'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.apply_atomized_element_names(batch)
  if knobs.get('restype_alignment'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.apply_restype_alignment_on_atomized(batch)
  if knobs.get('drop_atoms'):
    _drop_atoms_by_name(batch, knobs['drop_atoms'])
  if knobs.get('ccd_leaving_atoms'):
    _check_ccd_leaving(batch, fold_input, polymer_only=knobs['ccd_leaving_atoms'] == 'polymer')
  if knobs.get('std_conformers'):
    from alphafold3.constants import chai1_conformers
    _override_ref_conformers(batch, chai1_conformers.as_conformers())
  if knobs.get('raw_last_conformer'):
    _raw_last_conformer(batch)
  if knobs.get('esmfold2_ref_pos'):
    # ESMFold2's atom encoder reads its ccd.pkl conformers (see
    # constants/esmfold2_ref_pos.py), not the CCD ideal values AF3 uses.
    # Substituted by atom NAME through the same helper chai1's conformers use,
    # so an atom the table does not carry -- AF3's terminal OXT -- keeps the
    # coordinate it already had rather than being zeroed.
    from alphafold3.constants import esmfold2_ref_pos
    _override_ref_conformers(batch, esmfold2_ref_pos.as_conformers())
  if knobs.get('atom_keys_subset_size') and not os.environ.get(
      'AF3_NO_ESM_WIDE_KEYS'):
    # AF3_NO_ESM_WIDE_KEYS=1 is the A/B for this fidelity fix, in the style of
    # AF3_NO_ESM_REF_POS and AF3_NO_ESM_DROP_OXT -- the key subset is the only
    # atom-side feature that differs between esmfold2 and the models that place
    # an atomised entity correctly, so it has to be separable from them.
    _wide_key_window(batch, knobs['atom_keys_subset_size'])
  if knobs.get('qblock_keys'):
    _key_window(batch, 'slide_qblock')
  if knobs.get('circular_keys'):
    _circular_key_window(batch)
  if knobs.get('padded_keys'):
    _padded_key_window(batch)
  if knobs.get('chirals'):
    from alphafold3.model.pipeline import chiral_features
    chiral_features.attach_chiral_features(batch)
  if knobs.get('zero_msa_without_alignment') and not has_msa:
    _zero_msa(batch)
  if knobs.get('chai_msa') and has_msa and fold_input is not None:
    _chai_msa(batch, fold_input)
  if knobs.get('dna_gap_msa'):
    _dna_gap_msa(batch)
  if knobs.get('boltz_msa') and fold_input is not None and 'msa' in batch:
    _boltz_msa(batch, fold_input)
  if knobs.get('rf3_msa') and fold_input is not None and 'msa' in batch:
    _rf3_msa(batch, fold_input)
  if knobs.get('dedupe_self_msa'):
    _dedupe_self_msa(batch, query_once=knobs['dedupe_self_msa'] == 'query_once')
  if knobs.get('native_nonprotein_msa'):
    _native_nonprotein_msa(batch)
  if knobs.get('complex_msa_profile'):
    _complex_msa_profile(batch)
  if knobs.get('esm_deletion_mean'):
    _esm_deletion_mean(batch)
  if knobs.get('full_msa') or knobs.get('native_rng'):
    _crop_msa_padding(batch)
  if knobs.get('native_rng') and seed is not None and fold_input is not None:
    _native_rng(batch, seed, knobs['native_rng'], fold_input)
  if knobs.get('empty_template_gap'):
    _empty_template_gap(batch, slots=knobs.get('empty_template_gap_slots',
                                               'first'))
  if knobs.get('lm_pair') and lm_hidden is not None:
    _attach_lm_hidden(batch, *lm_hidden)
  elif knobs.get('lm_pair') and lm_pair is not None:
    _attach_lm_pair(batch, lm_pair)
  if knobs.get('esm') and esm is not None:
    _attach_esm(batch, esm)

  if knobs.get('opendde'):
    if refeaturise is None:
      raise ValueError('opendde needs refeaturise= to build its structural '
                       'token layout')
    from alphafold3.model.pipeline import structural_features
    batch = structural_features.attach_structural_batch(
        batch, refeaturise, struct_num_tokens=knobs.get('struct_num_tokens'))
    if knobs.get('padded_keys'):
      # the structural-token layout has its own atom windows, and it is the one
      # the diffusion runs on -- both need the convention.
      _padded_key_window(batch, prefix='struct/')
  return batch


def _modified_residue_positions(fold_input):
  """-> {(asym_id, 1-based residue index)} of every modified polymer residue.

  asym_id is 1-based and assigned in chain order, so chain i is asym_id i + 1.
  """
  positions = set()
  for index, chain in enumerate(fold_input.chains):
    mods = (getattr(chain, 'ptms', None)
            or getattr(chain, 'modifications', None) or ())
    for _code, residue in mods:
      positions.add((index + 1, int(residue)))
  return positions


def _mark_modified_residues(batch, fold_input, unknown_restype=False):
  """Add `is_modified`, and optionally give those tokens the unknown restype.

  Only Boltz-2 reads either, and it needs BOTH: its modified-residue conditioning
  and the unknown restype are the joint signature it was trained to read as "this
  token is a modified residue, take its shape from the reference conformer".
  Measured on phospho-ubiquitin with the residue kept as one token: restype alone
  leaves the phosphate 6.42 A out, the flag alone 6.53 A, and the two together
  3.10 A with a chemically correct 1.61 A OG-P bond. Every other family ignores
  both and keeps AlphaFold 3's convention, the PARENT restype -- which is why
  the restype change is opt-in rather than implied by there being modifications.

  Tokens are found by (chain, residue index) rather than by position: under
  AlphaFold 3's convention a modified residue is ATOMISED into one token per
  atom, so several tokens share one residue index.
  """
  from alphafold3.constants import residue_names

  wanted = _modified_residue_positions(fold_input)
  residue_index = np.asarray(batch['residue_index'])
  asym_id = np.asarray(batch['asym_id'])
  flag = np.zeros(residue_index.shape, np.int32)
  for token in range(residue_index.shape[0]):
    if (int(asym_id[token]), int(residue_index[token])) in wanted:
      flag[token] = 1
  batch['is_modified'] = flag
  if unknown_restype and flag.any():
    vocab = list(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
    unknown = vocab.index(residue_names.UNK)
    sel = flag.astype(bool)
    # genuine Boltz-2 (tokenize/boltz2.py, a modified residue tokenised at residue level): res_type is its CHAIN's
    # unknown, get_unk_token(chain) - a nucleic chain's N (our 30; a DNA chain's DN reads 30 too and model.py adds
    # Boltz's DN row on a DNA token). The protein unknown on a modified RNA base stretched 2'-O-methylguanosine's
    # O2'-CM2 to 2.6-2.7 A where genuine Boltz-2 holds 1.55-1.71.
    nucleic = np.zeros(flag.shape, bool)
    for k in ('is_rna', 'is_dna'):
      if k in batch:
        nucleic |= np.asarray(batch[k]).astype(bool).reshape(-1)
    aatype = np.array(batch['aatype'])
    aatype[sel & ~nucleic] = unknown
    aatype[sel & nucleic] = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP - 1
    batch['aatype'] = aatype
    if batch.get('msa') is not None:
      # boltz builds the query row from its tokens, so a modified residue reads the protein UNK there (and in the
      # query's copies), a nucleotide too; homolog rows keep their own letters
      msa, prof = np.array(batch['msa']), np.array(batch['profile'], np.float32)
      live = np.asarray(batch['msa_mask']).astype(bool)
      for chain in np.unique(asym_id[sel]):
        cols = np.flatnonzero(asym_id == chain)
        query = (msa[:, cols] == msa[0, cols]).all(-1) & live[:, cols].any(-1)
        for t in np.flatnonzero(sel & (asym_id == chain)):
          old = int(msa[0, t])
          n = max(int(live[:, t].sum()), 1)
          prof[t, old] -= query.sum() / n
          prof[t, unknown] += query.sum() / n
          msa[query, t] = unknown
      batch['msa'], batch['profile'] = msa, np.clip(prof, 0.0, 1.0).astype(np.asarray(batch['profile']).dtype)
    # ...and its representative atom is its first (parse_ccd_residue's atom_center=0, atom_disto=0)
    key = 'token_atoms_to_pseudo_beta'
    if f'{key}:gather_idxs' in batch:
      idxs = np.array(batch[f'{key}:gather_idxs'])
      mask = np.array(batch[f'{key}:gather_mask'])
      ref_mask = np.asarray(batch['ref_mask']).astype(bool)
      for t in np.nonzero(sel)[0]:
        on = np.nonzero(ref_mask[t])[0]
        if on.size:
          idxs[t] = t * ref_mask.shape[1] + int(on[0])
          mask[t] = True
      batch[f'{key}:gather_idxs'] = idxs
      batch[f'{key}:gather_mask'] = mask
  return batch


def cyclic_period(batch, chains=True, fold_input=None):
  """Wrap the relative-position encoding, so a chain has no N- or C-terminus.

  Not an AlphaFold 3 feature and not specific to any one model: the encoding is
  shared, so every family in this package honours it (see
  featurization.create_relative_encoding). A period of 0 means "not cyclic", so
  leaving this off is byte-identical to before.

  chains=True makes every polymer chain wrap at its own length; a sequence of
  chain ids wraps only those (fold_input maps the ids to asym_ids); a dict gives
  explicit per-chain periods, keyed by asym_id.

  The period is the chain's RESIDUE count, not its token count: a ligand
  contributes many tokens at one residue index, and wrapping on token count
  would misplace every polymer offset in the same chain.
  """
  asym_id = np.asarray(batch['asym_id'])
  residue_index = np.asarray(batch['residue_index'])
  period = np.zeros(asym_id.shape, np.int32)
  def wrap_at_own_length(a):
    selected = asym_id == a
    period[selected] = len(np.unique(residue_index[selected]))

  if chains is True:
    for a in np.unique(asym_id[asym_id > 0]):
      wrap_at_own_length(a)
  elif isinstance(chains, dict):
    for key, value in chains.items():
      period[asym_id == int(key)] = int(value)
  else:
    if fold_input is None:
      raise ValueError('naming cyclic chains by id needs fold_input=')
    # asym_id is 1-based and assigned in chain order.
    by_id = {chain.id: index + 1
             for index, chain in enumerate(fold_input.chains)}
    unknown = set(chains) - set(by_id)
    if unknown:
      raise ValueError(f'no chain(s) {sorted(unknown)} in this input; it has '
                       f'{sorted(by_id)}')
    for chain_id in chains:
      wrap_at_own_length(by_id[chain_id])
  batch['cyclic_period'] = period
  return batch


# Backbone atom pair that links residue i to residue i+1, per polymer class.
_LINK_ATOMS = (('C', 'N'), ("O3'", 'P'))


def _ccd_is_ligand(name, ccd_kind):
  """protenix get_mol_type: a residue whose CCD type is neither peptide (PEPTIDE-LIKE aside), DNA nor RNA is a ligand;
  PEPTIDE-LIKE (PNA) is left out, as native carries no links between those."""
  kind = ccd_kind(name)
  return bool(kind) and not ('PEPTIDE' in kind or 'DNA' in kind or 'RNA' in kind)


def _atomized_backbone_bonds(batch, fold_input=None, mode=True):
  """Restore the peptide bonds AlphaFold 3 drops at every atomised residue.

  AF3 extracts inter-residue bonds only where one side is a LIGAND chain
  (inter_chain_bonds.get_polymer_ligand_and_ligand_ligand_bonds), so a residue that
  is atomised INSIDE a polymer chain keeps its own CCD bonds and its declared
  covalent links but loses the backbone bond to each neighbour. AF3 does not miss
  them -- its relative-position encoding still places the atoms on one chain,
  because they share a residue index -- but atomworks emits them, so RF3 was trained
  with them present. On 5KX0, a 26-residue cyclic peptide with 11 atomised
  D-residues, native RF3's token_bonds carries 97 pairs and ours carried 83: every
  one of the 14 missing was a junction bond at an atomised residue.

  Bonds are written into the free (masked-off) rows of the ligand-ligand gather, so
  the batch keeps its shapes. Both directions are written: atomworks' matrix is
  symmetric, and AF3's featurisation gives only one direction.

  Not applied to the head-to-tail bond of a cyclic chain -- see cyclic_period, which
  is how every model in this package is told about that, and which is shared rather
  than RF3-specific.

  mode 'both_atomized': of3 -- token_bonds keeps a bond only when BOTH atoms are atomised (filter_fully_atomized_bonds), so
  a link counts between two consecutive non-standard residues and never to a standard neighbour.
  mode 'protenix': protenix's get_bond_features drops std-std, std-unstd and inter-unstd polymer bonds, so a link
  counts only where a partner's CCD type makes it a ligand (ACE, the PXZ chromophore), not a modified peptide.
  mode 'esm': esm's compute_token_bonds -- protein links only (prev C - N, C - next N) at an atomised residue, and a
  residue without that atom (PXZ, ACE, NH2) links through its first atomised token; no nucleic links, no CCD extras.
  """

  layout = batch.get('token_atoms_layout')
  if layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout

  ref_mask = np.asarray(batch['ref_mask']).astype(bool)
  num_token, num_dense = ref_mask.shape[-2], ref_mask.shape[-1]

  standard = _standard_residue_names()
  # residues in token order: (chain, res_id) -> [res_name, {atom_name: token}]
  residues = {}
  for t in range(num_token):
    for s in range(num_dense):
      if not ref_mask[t, s]:
        continue
      name = str(layout.atom_name[t, s])
      if not name:
        continue
      rkey = (str(layout.chain_id[t, s]), int(layout.res_id[t, s]))
      entry = residues.setdefault(rkey, [str(layout.res_name[t, s]), {}])
      entry[1].setdefault(name, t)

  from alphafold3.model import features as af3_features
  atomized = af3_features.ATOMIZED_RESIDUES
  ccd_kind = None
  if mode == 'protenix':
    from alphafold3.constants import chemical_components
    _ccd = chemical_components.Ccd(user_ccd=getattr(fold_input, 'user_ccd', None))
    ccd_kind = lambda name: ''.join((_ccd.get(name) or {}).get('_chem_comp.type', ())).upper()
  # a declared bond between two consecutive residues is their link (isoAsp joins through CG, not C)
  linked = {frozenset(((str(a[0]), int(a[1])), (str(b[0]), int(b[1]))))
            for a, b in (getattr(fold_input, 'bonded_atom_pairs', None) or ())}
  wanted = []
  keys = list(residues)
  for first, second in zip(keys, keys[1:]):
    if first[0] != second[0] or second[1] != first[1] + 1:
      continue                              # different chain, or a numbering gap
    (name_a, atoms_a), (name_b, atoms_b) = residues[first], residues[second]
    if name_a in standard and name_b in standard and first not in atomized and second not in atomized:
      continue                              # AF3 does not drop this one
    if frozenset((first, second)) in linked:
      continue
    if mode == 'both_atomized' and not ((name_a not in standard or first in atomized) and
                                        (name_b not in standard or second in atomized)):
      continue                              # of3 filter_fully_atomized_bonds: both partners atomised
    if mode == 'protenix' and not (_ccd_is_ligand(name_a, ccd_kind) or _ccd_is_ligand(name_b, ccd_kind)):
      continue                              # protenix drops polymer-polymer links: a partner must be a ligand by CCD type
    if mode == 'esm':
      prot = np.asarray(batch['is_protein']).astype(bool).reshape(-1)
      if not (prot[next(iter(atoms_a.values()))] and prot[next(iter(atoms_b.values()))]):
        continue
      lead = lambda atoms: min(atoms.values())
      wanted.append((atoms_a.get('C', lead(atoms_a)), atoms_b.get('N', lead(atoms_b))))
      continue
    for tail, head in _LINK_ATOMS:
      if tail in atoms_a and head in atoms_b:
        wanted.append((atoms_a[tail], atoms_b[head]))
        break
  # a backbone link is a single bond
  _add_token_bonds(batch, wanted, 1)
  if not atomized or mode in ('esm', 'both_atomized', 'protenix'):
    return batch
  # a standard residue atomised for its ligand bond (features.ATOMIZED_RESIDUES) has no bonds of its own in AF3's table:
  # its CCD bonds, by order
  from alphafold3.constants import chemical_components
  ccd = chemical_components.Ccd(user_ccd=getattr(fold_input, 'user_ccd', None))
  order_code = {'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 4}
  by_order = {}
  for rkey in atomized:
    if rkey not in residues:
      continue
    name, atoms = residues[rkey]
    entry = ccd.get(name) or {}
    for x, y, o in zip(entry.get('_chem_comp_bond.atom_id_1', ()), entry.get('_chem_comp_bond.atom_id_2', ()),
                       entry.get('_chem_comp_bond.value_order', ())):
      if x in atoms and y in atoms and atoms[x] != atoms[y]:
        by_order.setdefault(order_code.get(str(o).upper(), 1), []).append((atoms[x], atoms[y]))
  for code, pairs in by_order.items():
    _add_token_bonds(batch, pairs, code)
  return batch


def _ligand_bonded_units(batch, fold_input):
  """atomworks moves a residue it atomises for a ligand bond into the ligand's pn_unit: its tokens take the ligand's
  asym_id, keep their entity, and are the entity's next copy (sym_id)."""
  from alphafold3.model import features as af3_features
  atomized = af3_features.ATOMIZED_RESIDUES
  layout = batch.get('token_atoms_layout')
  if not atomized or layout is None:
    return batch
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  chain = layout.chain_id[:, 0].astype(str)
  res = layout.res_id[:, 0].astype(int)
  asym, entity, sym = (np.array(batch[k]) for k in ('asym_id', 'entity_id', 'sym_id'))
  asym_of = {c: int(asym[np.flatnonzero(chain == c)[0]]) for c in set(chain) if c}
  new_sym = {}
  for a, b in getattr(fold_input, 'bonded_atom_pairs', None) or ():
    poly, lig = (a, b) if (str(a[0]), int(a[1])) in atomized else (b, a)
    if (str(poly[0]), int(poly[1])) not in atomized or str(lig[0]) not in asym_of:
      continue
    toks = np.flatnonzero((chain == str(poly[0])) & (res == int(poly[1])))
    if not len(toks):
      continue
    e, unit = int(entity[toks[0]]), asym_of[str(lig[0])]
    if (e, unit) not in new_sym:
      new_sym[(e, unit)] = int(sym[entity == e].max()) + 1
    asym[toks], sym[toks] = unit, new_sym[(e, unit)]
  batch['asym_id'], batch['sym_id'] = asym, sym
  return batch


def _add_token_bonds(batch, pairs, order_code):
  """Token pairs, both directions, into the free (masked-off) rows of the ligand-ligand bond gather, so the batch keeps
  its shapes; grown if it has too few. order_code is the pairs' bond order."""
  key = 'tokens_to_ligand_ligand_bonds'
  idxs = np.array(batch[f'{key}:gather_idxs'])
  mask = np.array(batch[f'{key}:gather_mask'])
  order = batch.get('ligand_ligand_bond_order')
  present = {(int(a), int(b))
             for (a, b), ok in zip(idxs, mask.all(axis=1)) if ok}
  new = []
  for a, b in pairs:
    for pair in ((a, b), (b, a)):
      if pair not in present:
        present.add(pair)
        new.append(pair)
  if not new:
    return batch

  free = np.flatnonzero(~mask.all(axis=1))
  if free.size < len(new):
    # AF3 sizes this gather from the bond table, so a batch whose only bonds are
    # the ones we are adding arrives with no spare rows. Grow it rather than drop
    # bonds; the bond axis is not a bucketed dimension.
    grow = len(new) - free.size
    idxs = np.concatenate([idxs, np.zeros((grow, 2), idxs.dtype)])
    mask = np.concatenate([mask, np.zeros((grow, 2), mask.dtype)])
    if order is not None:
      order = np.concatenate([np.asarray(order),
                              np.zeros((grow,), np.asarray(order).dtype)])
    free = np.flatnonzero(~mask.all(axis=1))
  rows = free[:len(new)]
  idxs[rows] = np.asarray(new, idxs.dtype)
  mask[rows] = True
  batch[f'{key}:gather_idxs'] = idxs
  batch[f'{key}:gather_mask'] = mask
  if order is not None:
    order = np.array(order)
    order[rows] = order_code
    batch['ligand_ligand_bond_order'] = order
  return batch


def _declared_polymer_bonds(batch, fold_input, rule='all'):
  """bondedAtomPairs that AF3 does not featurise, as token bonds, by the vendor's own rule.

  AF3 keeps a declared bond only between a STANDARD residue or a ligand and a ligand; a bond between two polymer
  residues, or from a modified (atomised) residue to a ligand, never reaches its token bonds.

    'all'            boltz-2: its tokeniser makes every bond between two tokens a token bond (type + 1). A disulfide to
                     a D-Cys folded 21.7 A apart without it, against native's 2.4.
    'discontinuous'  protenix and opendde: `include_discont_poly_poly_bonds=True` at inference keeps a polymer-polymer
                     bond only between residues more than one apart or in different chains (head-to-tail ring closures,
                     staples, lactones), and every bond of a non-standard residue to a ligand.
    'ligand'         rf3: atomworks drops every bond between two non-atomised tokens, but keeps a modified (atomised)
                     residue's bonds, to a ligand or to another residue (BF6's backbone links).
  """
  from alphafold3.model import features

  pairs_in = getattr(fold_input, 'bonded_atom_pairs', None) if fold_input is not None else None
  layout = batch.get('token_atoms_layout')
  if not pairs_in or layout is None:
    return batch
  rule = 'all' if rule is True else rule
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  polymers = {c.id for c in fold_input.chains if type(c).__name__ in ('ProteinChain', 'RnaChain', 'DnaChain')}
  standard = _standard_residue_names()
  ref_mask = np.asarray(batch['ref_mask']).astype(bool)
  token_of, res_name = {}, {}
  for t, s in zip(*np.nonzero(ref_mask)):
    key = (str(layout.chain_id[t, s]), int(layout.res_id[t, s]), str(layout.atom_name[t, s]))
    token_of.setdefault(key, int(t))
    res_name[key[:2]] = str(layout.res_name[t, s])
  pairs = []
  for a, b in pairs_in:
    a, b = tuple(a), tuple(b)
    poly_a, poly_b = a[0] in polymers, b[0] in polymers
    if poly_a and poly_b:
      if rule == 'ligand' and all(res_name.get((str(p[0]), int(p[1]))) in standard for p in (a, b)):
        continue                            # rf3: kept only from an atomised (non-standard) residue
      if rule == 'discontinuous' and a[0] == b[0] and abs(int(a[1]) - int(b[1])) <= 1:
        continue                            # adjacent residues: protenix leaves the link to the chain
    elif poly_a or poly_b:
      poly = a if poly_a else b
      if res_name.get((str(poly[0]), int(poly[1]))) in standard:
        continue                            # AF3 featurises a standard residue's bond to a ligand itself
    else:
      continue                              # ligand-ligand: AF3's own
    ta, tb = token_of.get(a), token_of.get(b)
    if ta is not None and tb is not None and ta != tb:
      pairs.append((ta, tb))
  return _add_token_bonds(batch, pairs, features.BOND_ORDER_COVALENT)


def _check_ccd_leaving(batch, fold_input, polymer_only=False):
  """Warn if a declared bond's CCD leaving atom is still in the layout: the featuriser has to drop it before
  tokenising (featurise_input's ccd_leaving_atoms, which an older caller does not pass)."""
  from absl import logging
  from alphafold3.constants import chemical_components
  from alphafold3.model.pipeline import structure_cleaning

  pairs_in = getattr(fold_input, 'bonded_atom_pairs', None) if fold_input is not None else None
  layout = batch.get('token_atoms_layout')
  if not pairs_in or layout is None:
    return
  layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
  ref_mask = np.asarray(batch['ref_mask']).astype(bool)
  present, res_name = set(), {}
  for t, a in zip(*np.nonzero(ref_mask)):
    key = (str(layout.chain_id[t, a]), int(layout.res_id[t, a]))
    present.add(key + (str(layout.atom_name[t, a]),))
    res_name[key] = str(layout.res_name[t, a])
  ccd = chemical_components.Ccd(user_ccd=getattr(fold_input, 'user_ccd', None))
  from alphafold3.common import folding_input as fi
  ligand = {c.id for c in fold_input.chains if isinstance(c, fi.Ligand)} if polymer_only else set()
  left = sorted(k + (n,) for k, names in structure_cleaning.declared_bond_leaving_atoms(pairs_in, res_name, ccd).items()
                for n in names if k + (n,) in present and k[0] not in ligand)
  if left:
    logging.warning('declared-bond leaving atoms still present (featurised without ccd_leaving_atoms?): %s', left[:8])


def _declared_bond_feature(batch, fold_input):
  """chai-1's TokenBondRestraint: a (tokens, tokens) 0/1 matrix with [left, right] = 1 for each declared bond, every
  kind (polymer-polymer, residue-ligand, ligand-ligand), in its declared direction only; nothing else, not even a
  ligand's own bonds. chai-lab fills it from its covalent restraints and glycan strings (chai1.py,
  get_atom_covalent_bond_pairs_from_constraints) and projects it into both its trunk and its structure pair inputs."""
  num = np.asarray(batch['ref_mask']).shape[0]
  out = np.zeros((num, num), np.float32)
  pairs_in = getattr(fold_input, 'bonded_atom_pairs', None) if fold_input is not None else None
  layout = batch.get('token_atoms_layout')
  if pairs_in and layout is not None:
    layout = layout.item() if getattr(layout, 'ndim', None) == 0 else layout
    ref_mask = np.asarray(batch['ref_mask']).astype(bool)
    token_of = {}
    for t, s in zip(*np.nonzero(ref_mask)):
      token_of.setdefault((str(layout.chain_id[t, s]), int(layout.res_id[t, s]), str(layout.atom_name[t, s])), int(t))
    for a, b in pairs_in:
      ta, tb = token_of.get(tuple(a)), token_of.get(tuple(b))
      if ta is not None and tb is not None:
        out[ta, tb] = 1.0
  batch['declared_bonds'] = out
  return batch


def _standard_residue_names():
  from alphafold3.model.pipeline import chiral_features
  return chiral_features._standard_residues()
