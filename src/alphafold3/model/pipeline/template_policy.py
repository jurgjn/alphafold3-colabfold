"""How a model reads its templates: one policy, applied to the fold input before featurisation.

AF3's pipeline takes a template as given -- the mmCIF, the query->template index map -- and featurises it. The
vendors ported here each read a template differently, and every difference is a choice along one of five axes:

  axis        values                       who
  altloc      'occupancy' (AF3)            AF3, protenix, OpenDDE, IntelliFold
              'first'                      boltz (gemmi remove_alternative_conformations)
  modified    'parent' (AF3: SEP->SER)     AF3, IntelliFold
              'unk_n'                      boltz, protenix, OpenDDE: the residue is UNK with only its N --
                                           boltz atomises it and the query position takes its first atom's token;
                                           protenix maps it to UNK, whose dense atoms are all N
              'drop_hit'                   chai: a hit with any residue outside its standard codes is dropped
              'unk_restype'                OpenFold3 (0.4.5, v0.5.0): the residue keeps its own N/CA/C/CB, read by
                                           atom name, but its template restype is UNK (create_template_restype ->
                                           get_with_unknown_3_to_idx); MSE too, which its template path never turns
                                           into MET; one missing any of N/CA/C/CB is dropped (GAP)
                                           (map_token_pos_to_template_residues). A batch-level rule, after
                                           featurisation: unknown_restype_templates
  realign     'given' (AF3)                AF3, IntelliFold (re-aligns within the hit range), chai
              'whole_chain'                boltz (aligns the query to the whole template chain itself), protenix,
                                           OpenDDE (kalign to the chain's whole SEQRES), OpenFold3 (CIF-direct:
                                           kalign to the chain's canonical sequence)
  prefilter   False (AF3)
              True                         protenix, OpenDDE, IntelliFold (template_utils._assess_hit + get_templates):
                                           released after 2021-09-30, aligned to <= 10% of the query, a large
                                           duplicate (the hit a substring of the query covering > 95% of it -- a
                                           self-template never reaches these models), under 10 residues, or a
                                           repeat of an earlier hit's sequence
  hit_letter  'X' / 'parent'               the prefilter's hit sequence renders a modified residue as protenix's
                                           parser does ('X') or as IntelliFold's (its parent's letter); also the
                                           SEQRES whole_chain aligns to (OpenFold3: the canonical, parent letter)
  rank        False (AF3)                  templates in the order given
              True                         OpenFold3 (_parse_templates_from_cif_files): a template scoring
                                           seq_id * q_cov < 0.1 (cif_direct_min_score) is dropped, the rest are
                                           stable-sorted by that score, best first; both are per query residue

A policy is a dict of these keys (model_registry's `templates` featurise knob); a missing key keeps AF3's value, so an
empty policy is AF3's pipeline, byte for byte. Each rule was read off the vendor's code and checked against its own
featuriser on 5K9P (phospho-ubiquitin: a SEP, two altlocs), in vendor_native_parity's template / template_std /
multi_template cases.

One rule cannot live here: boltz gives an atomised (UNK) template residue no frame, which is a property of its
template features, not of its input -- template_modules.Boltz2TemplateEmbedding._features.
"""

import dataclasses
import re

import numpy as np

DEFAULT = dict(altloc='occupancy', modified='parent', realign='given', prefilter=False, hit_letter='X', rank=False)
_AXES = {'altloc': ('occupancy', 'first'), 'modified': ('parent', 'unk_n', 'drop_hit', 'unk_restype'),
         'realign': ('given', 'whole_chain'), 'prefilter': (False, True), 'hit_letter': ('X', 'parent'),
         'rank': (False, True)}

# OpenFold3 TemplatePreprocessorSettings.cif_direct_min_score
RANK_MIN_SCORE = 0.1

# protenix TemplateHitFeaturizer(max_template_date=...) at inference
PREFILTER_CUTOFF = '2021-09-30'

_STANDARD_AA = frozenset((
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'))
# chai_lab residue_constants.standard_residue_pdb_codes (MSE is NOT in it)
_CHAI_STANDARD_CODES = _STANDARD_AA | frozenset(('UNK', 'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT'))


def resolve(policy):
  """A policy dict -> every axis set, validated."""
  out = dict(DEFAULT, **(policy or {}))
  for k, v in out.items():
    if k not in _AXES or v not in _AXES[k]:
      raise ValueError('template policy %s=%r: want one of %s' % (k, v, _AXES.get(k)))
  return out


def is_default(policy):
  return resolve(policy) == DEFAULT


def apply(fold_input, policy):
  """`fold_input` with each protein chain's templates as the policy's model reads them."""
  from alphafold3.common import folding_input as fi
  p = resolve(policy)
  if p == DEFAULT:
    return fold_input
  chains = []
  for chain in fold_input.chains:
    if not isinstance(chain, fi.ProteinChain) or not chain.templates:
      chains.append(chain)
      continue
    query = getattr(chain, '_sequence', chain.sequence)
    kept, seen = [], set()
    for t in chain.templates:
      if p['modified'] == 'drop_hit' and not _residue_names(t.mmcif) <= _CHAI_STANDARD_CODES:
        continue
      qmap = dict(t.query_to_template_map)
      seqres = None
      if p['prefilter'] or p['realign'] == 'whole_chain' or p['rank']:
        seqres, date = template_seqres(t.mmcif, p['hit_letter'])
      if p['prefilter']:
        hit = ''.join(seqres[j] for j in sorted(qmap.values()) if j < len(seqres))
        if date and date > PREFILTER_CUTOFF:
          continue
        if len(qmap) / len(query) <= 0.1 or len(hit) < 10:
          continue
        if hit in query and len(hit) / len(query) > 0.95:
          continue
      if p['realign'] == 'whole_chain':
        qmap = global_align(query, seqres)
      if p['prefilter']:
        processed = ''.join(seqres[qmap[i]] for i in sorted(qmap))
        if processed in seen:
          continue
        seen.add(processed)
      cif = t.mmcif
      if p['altloc'] == 'first' or p['modified'] == 'unk_n':
        cif = rewrite_mmcif(cif, first_altloc=p['altloc'] == 'first', modified_unk_n=p['modified'] == 'unk_n')
      tpl = fi.Template(mmcif=cif, query_to_template_map=qmap)
      if p['rank']:
        # seq_id and q_cov are both over the query's length (compute_sequence_identity_and_coverage: every query
        # residue is a non-gap column of a global alignment)
        ident = sum(query[i] == seqres[j] for i, j in qmap.items() if j < len(seqres))
        score = ident * len(qmap) / len(query) ** 2
        if score < RANK_MIN_SCORE:
          continue
        tpl = (score, tpl)
      kept.append(tpl)
    if p['rank']:
      kept = [tpl for _, tpl in sorted(kept, key=lambda x: -x[0])]   # stable: ties keep their order
    # (ProteinChain and Template are slotted classes, not dataclasses: rebuilt)
    chains.append(fi.ProteinChain(
        id=chain.id, sequence=query, ptms=chain.ptms, description=chain.description,
        unpaired_msa=chain.unpaired_msa, paired_msa=chain.paired_msa, templates=kept))
  return dataclasses.replace(fold_input, chains=chains)


_UNK_RESTYPE, _GAP_RESTYPE = 20, 21   # our protein UNK and gap indices
# dense slots every standard amino acid shares (protein_data_processing): N, CA, C, CB
_BACKBONE_DENSE = (0, 1, 2, 4)


def unknown_restype_templates(batch, fold_input):
  """modified='unk_restype', on the featurised batch: each template residue outside the 20 standard amino acids
  (raw name: SEP, MSE, UNK, ...) reads restype UNK, with its N, CA, C and CB taken by atom name from the template
  itself into the dense slots every standard residue keeps them in (template_modules reads an UNK template residue's
  geometry from those slots for the models that use this); one missing any of the four is dropped to a gap.
  `fold_input` is the one template_policy.apply returned (its query->template maps are the ones featurised)."""
  from alphafold3 import structure
  from alphafold3.constants import atom_types
  from alphafold3.common import folding_input as fi
  if 'template_aatype' not in batch:
    return batch
  aatype = np.array(batch['template_aatype'])
  pos = np.array(batch['template_atom_positions'])
  mask = np.array(batch['template_atom_mask'])
  asym = np.asarray(batch['asym_id']).reshape(-1)
  resi = np.asarray(batch['residue_index']).reshape(-1)
  atom37 = [atom_types.ATOM37_ORDER[n] for n in ('N', 'CA', 'C', 'CB')]
  for ci, chain in enumerate(fold_input.chains):
    if not isinstance(chain, fi.ProteinChain) or not chain.templates:
      continue
    for k, t in enumerate(chain.templates[:aatype.shape[0]]):
      struc = structure.from_mmcif(t.mmcif, fix_mse_residues=False, fix_arginines=False, include_bonds=False,
                                   include_water=False, include_other=True).filter_to_entity_type(protein=True)
      names = next(iter(struc.chain_res_name_sequence(include_missing_residues=True).values()))
      arrays = struc.to_res_arrays(include_missing_residues=True, atom_order=atom_types.ATOM37_ORDER)
      for q, j in t.query_to_template_map.items():
        if j >= len(names) or names[j] in _STANDARD_AA:
          continue
        tok = np.nonzero((asym == ci + 1) & (resi == q + 1))[0]
        have = arrays.atom_mask[j][atom37] > 0
        pos[k, tok] = 0
        mask[k, tok] = 0
        if have.all():
          aatype[k, tok] = _UNK_RESTYPE
          for slot, a in zip(_BACKBONE_DENSE, atom37):
            pos[k, tok, slot] = arrays.atom_positions[j][a]
            mask[k, tok, slot] = 1
        else:
          aatype[k, tok] = _GAP_RESTYPE
  batch['template_aatype'] = aatype.astype(np.asarray(batch['template_aatype']).dtype)
  batch['template_atom_positions'] = pos.astype(np.asarray(batch['template_atom_positions']).dtype)
  batch['template_atom_mask'] = mask.astype(np.asarray(batch['template_atom_mask']).dtype)
  return batch


def _residue_names(cif):
  """The residue names present in a template's polymer chains, raw (MSE stays MSE)."""
  from alphafold3 import structure
  struc = structure.from_mmcif(cif, fix_mse_residues=False, fix_arginines=False, include_bonds=False,
                               include_water=False, include_other=True)
  return set(struc.filter_to_entity_type(protein=True, rna=True, dna=True, dna_rna_hybrid=True).res_name)


def template_seqres(cif, modified_letter='X'):
  """(one-letter SEQRES of the template's protein chain, its oldest revision date). A modified residue is 'X'
  (protenix's parser) or its CCD parent's letter (IntelliFold's); MSE is M either way."""
  from alphafold3 import structure
  from alphafold3.constants import residue_names
  struc = structure.from_mmcif(cif, fix_mse_residues=False, fix_arginines=False, include_bonds=False,
                               include_water=False, include_other=True)
  prot = struc.filter_to_entity_type(protein=True)
  names = next(iter(prot.chain_res_name_sequence(include_missing_residues=True).values()))
  one = residue_names.PROTEIN_COMMON_THREE_TO_ONE
  if modified_letter == 'parent':
    parents = next(iter(prot.chain_res_name_sequence(
        include_missing_residues=True, fix_non_standard_polymer_res=True).values()))
    seq = ''.join(one.get(n, one.get(pn, 'X')) for n, pn in zip(names, parents))
  else:
    seq = ''.join('M' if n == 'MSE' else one.get(n, 'X') for n in names)
  dates = (re.findall(r'(\d{4}-\d{2}-\d{2})', cif.split('_pdbx_audit_revision_history', 1)[1][:4000])
           if '_pdbx_audit_revision_history' in cif else [])
  return seq, (min(dates) if dates else '')


def global_align(a, b):
  """Needleman-Wunsch (match 1, mismatch -1, gap -1) of a onto b -> {i_a: i_b}. Stands in for the alignment a vendor
  re-aligns a template with (kalign for protenix, boltz's own): on the near-identical sequences a template is, the
  optimal global alignment is the one they return."""
  n, m = len(a), len(b)
  score = np.zeros((n + 1, m + 1), np.int32)
  score[:, 0] = -np.arange(n + 1)
  score[0, :] = -np.arange(m + 1)
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      score[i, j] = max(score[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1),
                        score[i - 1, j] - 1, score[i, j - 1] - 1)
  i, j, out = n, m, {}
  while i > 0 and j > 0:
    if score[i, j] == score[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1):
      out[i - 1] = j - 1
      i, j = i - 1, j - 1
    elif score[i, j] == score[i - 1, j] - 1:
      i -= 1
    else:
      j -= 1
  return out


def rewrite_mmcif(cif, first_altloc=False, modified_unk_n=False):
  """A template mmCIF with the policy's altloc and modified-residue rules applied, in a form AF3's template featuriser
  then reproduces. first_altloc: each residue keeps its FIRST altloc, where AF3 keeps the highest occupancy (5K9P: LYS
  29 and GLY 76 move 0.09 / 0.17 A). modified_unk_n: a modified amino acid (outside the 20 and MSE, which every vendor
  maps to MET as AF3 does) becomes an UNK residue holding only its N atom; AF3's dense layout for UNK is N throughout.
  Line-level on the _atom_site loop; every other category only has the residue names swapped."""
  lines = cif.splitlines()
  out, cols, i = [], [], 0
  modified = set()
  tok = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")
  while i < len(lines):
    line = lines[i]
    if line.startswith('_atom_site.'):
      while i < len(lines) and lines[i].startswith('_atom_site.'):
        cols.append(lines[i].split('.', 1)[1].strip())
        out.append(lines[i])
        i += 1
      c = {k: n for n, k in enumerate(cols)}
      rows = []
      while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '_', 'loop_')):
        rows.append(tok.findall(lines[i]))
        i += 1
      first_alt, seen_n = {}, set()
      for r in rows:
        if len(r) != len(cols):
          out.append(' '.join(r))
          continue
        key = (r[c['label_asym_id']], r[c['label_seq_id']], r[c.get('pdbx_PDB_ins_code', c['label_seq_id'])])
        alt = r[c['label_alt_id']]
        if first_altloc and alt not in ('.', '?'):
          if first_alt.setdefault(key, alt) != alt:
            continue
          r[c['label_alt_id']] = '.'
        comp = r[c['label_comp_id']]
        if modified_unk_n and r[c['group_PDB']] == 'HETATM' and comp not in _STANDARD_AA \
            and comp not in ('MSE', 'UNK') and r[c['label_seq_id']] not in ('.', '?'):
          modified.add(comp)
          if r[c['label_atom_id']] != 'N' or key in seen_n:
            continue
          seen_n.add(key)
          for k in ('label_comp_id', 'auth_comp_id'):
            if k in c:
              r[c[k]] = 'UNK'
          r[c['group_PDB']] = 'ATOM'
        out.append(' '.join(r))
      continue
    out.append(line)
    i += 1
  text = '\n'.join(out) + '\n'
  for comp in modified:
    text = re.sub(r'(?<=\s)%s(?=\s)' % re.escape(comp), 'UNK', text).replace('(%s)' % comp, '(UNK)')
  return text
