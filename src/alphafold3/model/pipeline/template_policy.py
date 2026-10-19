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
  realign     'given' (AF3)                AF3, IntelliFold (re-aligns within the hit range), chai
              'whole_chain'                boltz (aligns the query to the whole template chain itself), protenix,
                                           OpenDDE (kalign to the chain's whole SEQRES)
  prefilter   False (AF3)
              True                         protenix, OpenDDE, IntelliFold (template_utils._assess_hit + get_templates):
                                           released after 2021-09-30, aligned to <= 10% of the query, a large
                                           duplicate (the hit a substring of the query covering > 95% of it -- a
                                           self-template never reaches these models), under 10 residues, or a
                                           repeat of an earlier hit's sequence
  hit_letter  'X' / 'parent'               the prefilter's hit sequence renders a modified residue as protenix's
                                           parser does ('X') or as IntelliFold's (its parent's letter)

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

DEFAULT = dict(altloc='occupancy', modified='parent', realign='given', prefilter=False, hit_letter='X')
_AXES = {'altloc': ('occupancy', 'first'), 'modified': ('parent', 'unk_n', 'drop_hit'),
         'realign': ('given', 'whole_chain'), 'prefilter': (False, True), 'hit_letter': ('X', 'parent')}

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
      if p['prefilter'] or p['realign'] == 'whole_chain':
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
      kept.append(fi.Template(mmcif=cif, query_to_template_map=qmap))
    # (ProteinChain and Template are slotted classes, not dataclasses: rebuilt)
    chains.append(fi.ProteinChain(
        id=chain.id, sequence=query, ptms=chain.ptms, description=chain.description,
        unpaired_msa=chain.unpaired_msa, paired_msa=chain.paired_msa, templates=kept))
  return dataclasses.replace(fold_input, chains=chains)


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
