"""The STOCK HelixFold3 (PaddlePaddle) half of vendor_native_parity.py (helixfold3).

  python helixfold3_upstream_runner.py featurise  helixfold3 CASE.json OUT_DIR
  python helixfold3_upstream_runner.py forward    helixfold3 OUT_DIR IN.npz OUT.npz
  python helixfold3_upstream_runner.py trajectory helixfold3 OUT_DIR IN.npz OUT.npz

Run in ~/venv_helixfold3 (paddle 3.1) against ~/PaddleHelix/apps/protein_folding/helixfold3
(HF3_ROOT). Everything is HF3's own, called the way its `inference.py:main` calls it:
`preprocess_json_to_entity` (schema validation + entity parsing), `get_msa_templates_pipeline`
(use_precomputed_msas=True), `feature_processing_aa.featurize_entities`, `batch_convert`,
`config.model_config('allatom_demo')` + `RunModel` + the 20250714 checkpoint, and the
model's own forward. The only overrides are monkeypatches that wrap/record/pass through:

  * the DATABASE SEARCHES. HF3 always reads its MSAs from precomputed files under
    <out>/msas/<type>_<chain>/<chain>/ (inference.py hardcodes use_precomputed_msas=True), so
    the case's MSAs are WRITTEN as those files (Stockholm, see `_write_msas`). The template
    search (hmmsearch over pdb_seqres, never cached by HF3) is the one tool call replaced:
    `Hmmsearch.query` returns the case's templates as the hmmsearch Stockholm hit list HF3
    would have got; HF3's own HmmsearchHitFeaturizer then parses the mmCIF (offline dir),
    kalign-realigns, prefilters and featurises them. Dummy DB paths are an empty file.
  * `forward`: DiffusionModule.sample_diffusion returns the injected positions and runs the
    real denoiser (`_forward_model`) once on the injected (noisy, sigma); paddle.randperm
    (MSA row draw) and CentreRandomAugmentation (confidence-head augmentation) are wrapped
    to RECORD their draws.
  * `trajectory`: inside sample_diffusion only, paddle.normal / scipy Rotation.random are
    replaced by the harness's draws.

TF32: paddle allows TF32 cuBLAS by default on Ampere (core.get_cublas_switch() is True);
switched OFF here (core.set_cublas_switch(False), NVIDIA_TF32_OVERRIDE=0). HF3_TF32=1 keeps it.

CONVENTIONS a consumer must know (also printed into OUT_DIR/hf3_meta.json):
  * TOKEN ORDER. HF3 reorders chains: protein -> rna (MSA pairing may regroup by sequence)
    -> dna -> ligand. Every TOKEN-axis array this runner emits (layout atom_token, s_inputs,
    s, z, pae*, pde*, feat.* token features) is permuted back to the CASE's chain order
    (layout `hf3_token_perm`: case token k = HF3 token hf3_token_perm[k]).
  * FLAT ATOM ORDER is HF3's own (its chain order, then per residue the CCD order of the
    preprocessed CCD pkl), and contains EVERY atom of each CCD entry: OXT on every amino
    acid, OP3 on every nucleotide, leaving atoms of modified residues/ligands. positions /
    denoised / plddt are in this order; atom_token maps each to its CASE token index.
  * MSA rows: msa_rows_pass{i} is the ORDERED index array paddle.randperm(n_rows)[:msa_depth]
    of pass i (i = 0..recycling_steps) into the row axis of HF3's merged MSA (layout
    feat.msa, rows in HF3's order: paired block first, then unpaired). MSA_ALL_ROWS lifts
    msa_depth (4096) to all rows; the permutation is still drawn and recorded.
  * recycling_steps: HF3 runs range(1 + num_recycle) trunk passes; num_recycle :=
    recycling_steps (stock 9), as protenix's N_cycle = recycling_steps + 1.
"""
import json
import os
import pickle
import re
import sys

if not os.environ.get('HF3_TF32'):
  os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

import numpy as np

# HF3's requirements.txt leaves numpy unpinned and ~/venv_helixfold3 has 1.24, which removed
# the `np.object` alias its template featuriser's zero-hit branch uses (templates.py): an
# environment shim restoring the alias numpy < 1.24 had, nothing of the model.
if not hasattr(np, 'object'):
  np.object = object

HF3_ROOT = os.path.expanduser(os.environ.get(
    'HF3_ROOT', '~/PaddleHelix/apps/protein_folding/helixfold3'))
CKPT = os.path.expanduser(os.environ.get(
    'HF3_CKPT', '~/helixfold3_params/HelixFold3-params-20250714/HelixFold3-20250714.pdparams'))
CCD = os.path.expanduser(os.environ.get('HF3_CCD', '~/helixfold3_data/ccd_preprocessed_etkdg.pkl.gz'))
MODEL_NAME = 'allatom_demo'
SEED = 1
sys.path.insert(0, HF3_ROOT)


def _vendor_na(msg):
  """The stock vendor cannot express this input: exit 3, graded N/A by the harness."""
  print('VENDOR_NA: ' + msg, file=sys.stderr)
  sys.exit(3)


def _int_id_to_str_id(num):
  out = []
  num -= 1
  while num >= 0:
    out.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(out)


# ------------------------------------------------------------------ input translation

def _hf3_json(j):
  """AF3 fold-input JSON -> (HF3 input JSON, [(case chain id, HF3 chain id, kind, entry)])."""
  if j.get('bondedAtomPairs'):
    # HF3's featuriser passes coval_bonds_info=[] (feature_processing_aa.py) and its JSON
    # schema has no bond field: there is no way to give it a user bond.
    _vendor_na('bondedAtomPairs: HF3 inference takes no covalent bond input')
  if j.get('userCCD') or j.get('userCCDPath'):
    _vendor_na('userCCD: HF3 reads components only from its preprocessed CCD pickle')
  ents, chains = [], []
  for k, entry in enumerate(j['sequences'], start=1):
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    for s, cid in enumerate(ids, start=1):
      chains.append((cid, '%s-%d' % (_int_id_to_str_id(k), s), kind, e))
    if kind in ('protein', 'rna', 'dna'):
      d = {'type': kind, 'sequence': e['sequence'], 'count': len(ids)}
      mods = [{'type': 'residue_replace', 'ccd': m['ptmType'], 'index': m['ptmPosition']}
              if kind == 'protein' else
              {'type': 'residue_replace', 'ccd': m['modificationType'], 'index': m['basePosition']}
              for m in e.get('modifications', [])]
      if mods:
        d['modification'] = mods
      if kind == 'rna' and e.get('modifications'):
        # verified: stock HF3 featurises a modified RNA residue as restype N (ccd_to_restype)
        # while its MSA row 0 keeps the parent base, and its own check_chain_orders_of_seq_and_msa
        # then raises 'The chain order of seq and msa is wrong' (both MSA modes)
        _vendor_na('modified RNA residue: stock HF3 featurisation asserts on it')
      if kind != 'protein' and e.get('templates'):
        _vendor_na('%s templates: HF3 has protein templates only' % kind)
      ents.append(d)
    elif kind == 'ligand':
      if 'smiles' in e:
        ents.append({'type': 'ligand', 'smiles': e['smiles'], 'count': len(ids)})
      else:
        codes = list(e['ccdCodes'])
        if len(codes) != 1:
          # an HF3 ligand entity is ONE CCD component (`"ccd": "ATP"`); a multi-component
          # ligand (a glycan chain) is only expressible with the bonds it does not take
          _vendor_na('multi-component ligand %s: HF3 ligands are one CCD code each' % codes)
        # AF3 has no ion type: an ion is a CCD ligand. HF3's `ion` and `ligand` entities go
        # through the same ligand_convert (dtype 'ligand'), so `ligand` is used for both.
        ents.append({'type': 'ligand', 'ccd': codes[0], 'count': len(ids)})
    else:
      _vendor_na('entity kind %r' % kind)
  return {'entities': ents}, chains


def _a3m_rows(a3m):
  rows, desc, seq = [], None, []
  for line in (a3m or '').splitlines():
    line = line.strip()
    if not line:
      continue
    if line.startswith('>'):
      if desc is not None:
        rows.append((desc, ''.join(seq)))
      desc, seq = line[1:].strip(), []
    else:
      seq.append(line)
  if desc is not None:
    rows.append((desc, ''.join(seq)))
  return rows


def _split_a3m(seq):
  """a3m row -> (match states (uppercase/'-'), insertion string before each match + after)."""
  match, ins, cur = [], [], []
  for c in seq:
    if c.islower() or c == '.':
      if c != '.':
        cur.append(c)
    else:
      ins.append(''.join(cur))
      cur = []
      match.append(c)
  ins.append(''.join(cur))
  return match, ins


def _sto(rows, pad):
  """a3m rows -> Stockholm text, insertion columns materialised (padded with `pad`).

  HF3's parse_stockholm takes the FILE's first row as the query frame (columns where it has
  '-' are deletions, counted per row as non-'-' residues), so protein pads with '-'.
  parse_stockholm_RNA prepends the input sequence and finds the insertion columns from the
  file's first row (lowercase or '.'), so RNA pads with '.'."""
  if not rows:
    return '# STOCKHOLM 1.0\n//\n'
  split = [_split_a3m(s) for _, s in rows]
  L = len(split[0][0])
  for (d, _), (m, _) in zip(rows, split):
    if len(m) != L:
      raise ValueError('a3m row %r has %d match states, query %d' % (d, len(m), L))
  width = [max(len(ins[g]) for _, ins in split) for g in range(L + 1)]
  out = ['# STOCKHOLM 1.0', '']
  for k, ((d, _), (m, ins)) in enumerate(zip(rows, split)):
    parts = []
    for g in range(L + 1):
      parts.append(ins[g] + pad * (width[g] - len(ins[g])))
      if g < L:
        parts.append(m[g])
    # unique names (Stockholm rows are keyed by name); the '/k' suffix is what jackhmmer
    # appends (/start-end) and msa_identifiers strips before parsing the UniProt species
    name = re.sub(r'\s', '_', d.split()[0] if d.split() else 'seq') + '/%d' % k
    out.append('%s %s' % (name, ''.join(parts)))
  # the reference annotation jackhmmer/nhmmer write (match 'x', insert '.'): HF3's
  # remove_empty_columns_from_stockholm_msa processes the alignment chunk by chunk at it
  out.append('#=GC RF %s' % ''.join('.' * width[g] + ('x' if g < L else '') for g in range(L + 1)))
  return '\n'.join(out) + '\n//\n'


def _msa_nontrivial(e):
  rows = _a3m_rows(e.get('unpairedMsa'))
  return len(rows) > 1 or len(_a3m_rows(e.get('pairedMsa'))) > 1 or bool(e.get('templates'))


def _write_msas(chains, msa_dir):
  """The case's MSAs as HF3's precomputed search output, for the first chain of each
  distinct sequence (HF3 searches once per sequence, featurize_entities):

   protein  <msa_dir>/protein_<C>/<C>/{uniref90,reduced_bfd,mgnify,uniprot}_hits.sto
            unpairedMsa -> uniref90 (+ reduced_bfd, mgnify as overflow: monomer_process
            reads them truncated to 10000/5000/5000 rows and concatenates uniref90, bfd,
            mgnify, deduplicated -- each file restates the query as its first row);
            pairedMsa -> uniprot_hits.sto, read ONLY for cross-chain pairing (*_all_seq):
            HF3 pairs by the UniProt species in the header, as AF3 does. Rows of a paired
            MSA are NOT added to the unpaired MSA, and a monomer/homomer drops them.
   rna      <msa_dir>/rna_<C>/<C>/rfam_hits.sto (HF3 prepends the query itself); the same
            rows also feed RNA pairing (rfam is HF3's RNA pairing source).
  """
  seen = set()
  for cid, hid, kind, e in chains:
    if kind not in ('protein', 'rna') or e['sequence'] in seen:
      continue
    seen.add(e['sequence'])
    d = os.path.join(msa_dir, '%s_%s' % (kind, hid), hid)
    os.makedirs(d, exist_ok=True)
    q = e['sequence']
    rows = _a3m_rows(e.get('unpairedMsa')) or [('query', q)]
    if kind == 'protein':
      if rows[0][1].replace('-', '').upper() != q:
        rows = [('query', q)] + rows
      query, rest = rows[0], rows[1:]
      chunks = [('uniref90_hits.sto', 9999), ('reduced_bfd_hits.sto', 4999), ('mgnify_hits.sto', 4999)]
      for fname, n in chunks:
        with open(os.path.join(d, fname), 'w') as f:
          f.write(_sto([query] + rest[:n], '-'))
        rest = rest[n:]
      if rest:
        print('WARNING: %d unpaired rows past HF3\'s 20000-row precomputed capacity dropped' % len(rest),
              file=sys.stderr)
      paired = _a3m_rows(e.get('pairedMsa'))
      if not paired or paired[0][1].replace('-', '').upper() != q:
        paired = [('query', q)] + paired
      with open(os.path.join(d, 'uniprot_hits.sto'), 'w') as f:
        f.write(_sto(paired, '-'))
    else:
      if _a3m_rows(e.get('pairedMsa')):
        print('WARNING: RNA pairedMsa ignored (HF3 pairs RNA from its rfam rows)', file=sys.stderr)
      # drop the restated query only when it is row 0 (HF3 prepends its own)
      hits = rows[1:] if rows and rows[0][1].replace('-', '').upper() == q.upper() else rows
      with open(os.path.join(d, 'rfam_hits.sto'), 'w') as f:
        f.write(_sto(hits, '.') if hits else '')


def _template_hits(chains, tdir):
  """Each protein chain's AF3 templates as the hmmsearch Stockholm HF3 would have got back
  (`>pdbid_chain/start-end [subseq from] mol:protein length:N`, the hit row aligned to the
  query: mapped template residues uppercase, unmapped template residues between them
  lowercase, unmapped query positions '-'), the mmCIF written to the offline mmcif dir.
  -> {query sequence: sto}"""
  from helixfold.data import mmcif_parsing
  mdir = os.path.join(tdir, 'mmcif')
  os.makedirs(mdir, exist_ok=True)
  out = {}
  for cid, hid, kind, e in chains:
    if kind != 'protein' or not e.get('templates') or e['sequence'] in out:
      continue
    q = e['sequence']
    lines = ['# STOCKHOLM 1.0', '']
    rows = []
    for t in e['templates']:
      cif = t['mmcif']
      pdb_id = re.search(r'_entry\.id\s+(\S+)', cif).group(1).lower()
      with open(os.path.join(mdir, '%s.cif' % pdb_id), 'w') as f:
        f.write(cif)
      res = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=cif)
      if res.mmcif_object is None:
        raise RuntimeError('HF3 cannot parse template %s: %s' % (pdb_id, res.errors))
      m = re.search(r'_entity_poly\.pdbx_strand_id\s+(\S+)', cif)
      chain = (m.group(1).split(',')[0] if m else '').strip("'\"")
      if chain not in res.mmcif_object.chain_to_seqres:
        chain = next(iter(res.mmcif_object.chain_to_seqres))
      seqres = res.mmcif_object.chain_to_seqres[chain]
      qmap = dict(zip(t['queryIndices'], t['templateIndices']))
      hit, last = [], None
      for i in range(len(q)):
        if i in qmap:
          j = qmap[i]
          if last is not None:
            hit.append(seqres[last + 1:j].lower())
          hit.append(seqres[j])
          last = j
        else:
          hit.append('-')
      ts = sorted(qmap.values())
      name = '%s_%s/%d-%d' % (pdb_id, chain, ts[0] + 1, ts[-1] + 1)
      rows.append((name, ''.join(hit), len(seqres)))
    for k, (name, _, n) in enumerate(rows):
      # hmmsearch names are unique per domain; a second hit of the same chain differs in range
      lines.append('#=GS %s DE [subseq from] mol:protein length:%d  case template %d' % (name, n, k))
    lines.append('')
    width = max(len(h) for _, h, _ in rows)
    for name, h, _ in rows:
      lines.append('%s %s' % (name, h + '.' * (width - len(h))))
    out[q] = '\n'.join(lines) + '\n//\n'
  return out


# ------------------------------------------------------------------ HF3 featurisation

def _args(out_dir):
  import argparse
  dummy = os.path.join(out_dir, 'db', 'empty.db')
  os.makedirs(os.path.dirname(dummy), exist_ok=True)
  open(dummy, 'a').close()
  mdir = os.path.join(out_dir, 'templates', 'mmcif')
  os.makedirs(mdir, exist_ok=True)
  if not any(n.endswith('.cif') for n in os.listdir(mdir)):
    # HmmsearchHitFeaturizer refuses an mmcif dir with no CIFs at construction; a
    # template-less case's dir holds this one file, which no hit names
    open(os.path.join(mdir, '_no_templates.cif'), 'w').close()
  import shutil
  return argparse.Namespace(
      preset='reduced_dbs', reduced_bfd_database_path=dummy, uniref90_database_path=dummy,
      mgnify_database_path=dummy, uniprot_database_path=dummy, pdb_seqres_database_path=dummy,
      rfam_database_path=dummy, template_mmcif_dir=mdir,
      # AF3 takes a user template whatever its date; no cutoff here either
      max_template_date='9999-12-31', obsolete_pdbs_path=None,
      jackhmmer_binary_path='/nonexistent/jackhmmer', hhblits_binary_path='/nonexistent/hhblits',
      hhsearch_binary_path='/nonexistent/hhsearch', hmmsearch_binary_path='/nonexistent/hmmsearch',
      hmmbuild_binary_path='/nonexistent/hmmbuild', nhmmer_binary_path='/nonexistent/nhmmer',
      kalign_binary_path=shutil.which('kalign') or os.path.expanduser('~/.local/bin/kalign'))


def _hf3_featurise(out_dir, use_msa, template_sto):
  import pathlib
  import inference as I
  from helixfold.data.tools import hmmsearch
  from infer_scripts import feature_processing_aa
  from helixfold.data import pipeline_aa_utils

  def query(self, msa_sto):
    # the query is the first row of the (deduplicated, empty-column-stripped) uniref90 MSA
    first = next(l for l in msa_sto.splitlines() if l.strip() and not l.startswith(('#', '//')))
    seq = first.split()[1].replace('-', '').replace('.', '').upper()
    return template_sto.get(seq, '# STOCKHOLM 1.0\n//\n')
  hmmsearch.Hmmsearch.query = query
  I.init_seed(SEED)
  base = pathlib.Path(out_dir)
  entities = I.preprocess_json_to_entity(os.path.join(out_dir, 'input', 'hf3.json'), base)
  bad = [e.error_message for e in entities if e.error_code]
  if bad:
    raise RuntimeError('HF3 rejected the input: %s' % bad)
  pipes = I.get_msa_templates_pipeline(_args(out_dir)) if use_msa else {}
  # featurize_entities runs the per-chain MSA tasks in a ProcessPoolExecutor (fork): the
  # patched Hmmsearch.query is inherited
  return feature_processing_aa.featurize_entities(
      entities, ccd_preprocessed_path=CCD, msa_templ_data_pipeline_dict=pipes,
      msa_output_dir=base.joinpath('msas'), use_msa_templ_feats=use_msa)


def featurise(model, case_json, out_dir):
  assert model == 'helixfold3', model
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  hj, chains = _hf3_json(j)
  if any(e.get('cyclic') for _, _, _, e in chains):
    _vendor_na('cyclic polymer: HF3 has no cyclic input')
  os.makedirs(os.path.join(out_dir, 'input'), exist_ok=True)
  with open(os.path.join(out_dir, 'input', 'hf3.json'), 'w') as f:
    json.dump(hj, f, indent=1)
  # Stock HF3 has two MSA modes: the pipeline (precomputed search files) or NO_MSA=1, where
  # every chain gets a query-only MSA and MOCKED templates. A case whose every protein/RNA
  # chain is query-only with no templates is the NO_MSA run; anything else the pipeline,
  # query-only files for the chains without one. (The two differ in the empty-template
  # features: pipeline gives HF3's zero hit, NO_MSA its gap-aatype mock.)
  use_msa = any(_msa_nontrivial(e) for _, _, kind, e in chains if kind in ('protein', 'rna'))
  template_sto = {}
  if use_msa:
    _write_msas(chains, os.path.join(out_dir, 'msas'))
    template_sto = _template_hits(chains, os.path.join(out_dir, 'templates'))
  for root, _, files in os.walk(os.path.join(out_dir, 'msas')):
    for fn in files:
      if fn == 'features.pkl':   # HF3's per-chain cache: never a stale one
        os.remove(os.path.join(root, fn))
  sample = _hf3_featurise(out_dir, use_msa, template_sto)
  feat, label = sample['feat'], sample['label']

  # chain bookkeeping: HF3 token -> chain, case chain order -> token permutation
  tok_chain = np.array(str(feat['chain_ids']).split())
  n_tok = len(tok_chain)
  perm = np.concatenate([np.flatnonzero(tok_chain == hid) for _, hid, _, _ in chains])
  assert sorted(perm.tolist()) == list(range(n_tok)), 'chain bookkeeping lost tokens'
  inv = np.empty_like(perm)
  inv[perm] = np.arange(n_tok)
  atom_tok_hf3 = np.asarray(feat['ref_token2atom_idx'])
  names = str(feat['all_atom_ids']).split()
  assert len(names) == len(atom_tok_hf3)
  meta = {'use_msa': bool(use_msa), 'hf3_token_perm': perm.tolist(),
          'chains': [[cid, hid, kind] for cid, hid, kind, _ in chains],
          'hf3_chain_order': list(dict.fromkeys(tok_chain.tolist())),
          'n_token': int(n_tok), 'n_atom': len(names), 'msa_rows': int(np.shape(feat['msa'])[0])}
  with open(os.path.join(out_dir, 'hf3_meta.json'), 'w') as f:
    json.dump(meta, f, indent=1)
  with open(os.path.join(out_dir, 'hf3_batch.pkl'), 'wb') as f:
    pickle.dump({'feat': feat, 'label': label, 'meta': meta}, f, protocol=4)

  tok = lambda k: np.asarray(feat[k])[perm]
  extra = {'feat.' + k: np.asarray(feat[k])[:, perm].astype(np.float32)
           for k in ('msa', 'has_deletion', 'deletion_value') if k in feat}
  extra.update({'feat.' + k: tok(k).astype(np.float32) for k in ('profile', 'deletion_mean')})
  extra.update({'feat.' + k: np.asarray(feat[k])[:, perm] for k in feat
                if k.startswith('template_') and np.ndim(feat[k]) >= 2
                and np.shape(feat[k])[1] == n_tok and np.asarray(feat[k]).dtype.kind != 'O'})
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names),
           atom_token=inv[atom_tok_hf3],
           atom_pad=np.ones(len(names), bool),
           token_pad=np.ones(n_tok, bool),
           hf3_token_perm=perm,
           **{'feat.ref_pos': np.asarray(feat['ref_pos'], np.float32),
              'feat.ref_space_uid': np.asarray(feat['ref_space_uid']),
              'feat.ref_element': np.asarray(feat['ref_element']),
              'feat.ref_charge': np.asarray(feat['ref_charge']),
              'feat.restype': tok('restype'),
              'feat.residue_index': tok('residue_index'),
              'feat.asym_id': tok('asym_id'), 'feat.entity_id': tok('entity_id'),
              'feat.sym_id': tok('sym_id'),
              'feat.token_bonds': np.asarray(feat['token_bonds'])[perm][:, perm]},
           **extra)


# ------------------------------------------------------------------ model

def _no_bf16_bias():
  """Diagnostic (HF3_NO_BF16_BIAS): stock's Attention without its bfloat16 cast of the
  triangle pair bias, so a run can be compared with no rounding boundary in it."""
  import inspect, textwrap
  from helixfold.model import modules
  src = textwrap.dedent(inspect.getsource(modules.Attention.forward))
  new = src.replace('.cast("bfloat16")', '')
  assert new != src
  ns = {}
  exec(compile(new, modules.__file__, 'exec'), modules.__dict__, ns)
  modules.Attention.forward = ns['forward']


def _setup(out_dir, recycling_steps, diff_batch=1):
  import paddle
  if os.environ.get('HF3_NO_BF16_BIAS'):
    _no_bf16_bias()
  from paddle.base import core
  if not os.environ.get('HF3_TF32'):
    core.set_cublas_switch(False)
  import inference as I
  from helixfold.model import config
  from utils.model import RunModel
  with open(os.path.join(out_dir, 'hf3_batch.pkl'), 'rb') as f:
    saved = pickle.load(f)
  I.init_seed(SEED)
  cfg = config.model_config(MODEL_NAME)
  cfg.model.num_recycle = int(recycling_steps)
  cfg.model.heads.diffusion_module.test_diff_batch_size = diff_batch
  n_rows = int(np.shape(saved['feat']['msa'])[0])
  if os.environ.get('MSA_ALL_ROWS') and n_rows > cfg.model.embeddings_and_pairformer.msa_module.msa_depth:
    # MsaModule takes randperm(rows)[:msa_depth=4096] each pass: a draw the harness lifts
    cfg.model.embeddings_and_pairformer.msa_module.msa_depth = n_rows
  model = RunModel(cfg)
  pd = paddle.load(CKPT)
  missing, unexpected = model.helixfold.set_state_dict(pd['model'] if 'optimizer' in pd else pd) or ([], [])
  model.eval()
  batch = {'feat': I.batch_convert(dict(saved['feat']), add_batch=True),
           'label': I.batch_convert(dict(saved['label']), add_batch=True),
           'label_cropped': {}}
  return model, batch, saved['meta'], cfg


def _run(model, batch, precision):
  import paddle
  from utils.utils import get_custom_amp_list
  with paddle.no_grad():
    if precision == 'bf16':
      black, white = get_custom_amp_list()
      with paddle.amp.auto_cast(enable=True, custom_white_list=white, custom_black_list=black,
                                level='O1', dtype='bfloat16'):
        return model(batch, compute_loss=False)
    return model(batch, compute_loss=False)


def _np(x):
  import paddle
  if isinstance(x, paddle.Tensor):
    if x.dtype == paddle.bfloat16:
      x = x.cast('float32')
    return np.array(x.numpy())
  return np.asarray(x)


def forward(model_name, out_dir, in_npz, out_npz):
  import paddle
  from helixfold.model import diffusion
  from helixfold.common import confidence
  z = np.load(in_npz)
  precision = 'bf16' if str(z['precision']) == 'bf16' else 'fp32'
  model, batch, meta, cfg = _setup(out_dir, int(z['recycling_steps']))
  net = model.helixfold
  perm = np.asarray(meta['hf3_token_perm'])
  pos = paddle.to_tensor(z['positions'], dtype='float32')
  noisy = paddle.to_tensor(z['positions_noisy'], dtype='float32')
  sigma = float(z['noise_level'])
  got, rows = {}, []
  dm = net.diffusion_module

  def sample_diffusion(representations, batch, step_num=None, gamma0=None):
    atom_mask = batch['all_atom_pos_mask']
    B = atom_mask.shape[0]
    assert B == 1 and pos.shape[0] == atom_mask.shape[1], (pos.shape, atom_mask.shape)
    den = dm._forward_model(noisy[None], paddle.full([1], sigma, dtype='float32'), batch, representations)
    got['denoised'] = _np(den)[0]
    if dm.enable_diff_cache:
      dm.clear_cache()
    return {'final_atom_positions': pos[None] * atom_mask[..., None], 'final_atom_mask': atom_mask}
  dm.sample_diffusion = sample_diffusion

  def keep(name):
    def hook(_layer, _inp, out):
      got[name] = [_np(o) for o in out] if isinstance(out, (tuple, list)) else _np(out)
    return hook
  net.input_embedder.register_forward_post_hook(keep('input_embedder'))
  if os.environ.get('VENDOR_NUDGE'):
    # the harness's CASE_FLOOR: the vendor against ITSELF, its initial pair nudged by a relative VENDOR_NUDGE
    eps = float(os.environ['VENDOR_NUDGE'])

    def nudge(_layer, _inp, out):
      s_in, s_init, z_init = out
      noise = paddle.to_tensor(np.random.default_rng(int(os.environ.get('VENDOR_NUDGE_SEED', 7))).normal(size=z_init.shape).astype('float32'))
      return s_in, s_init, z_init + eps * float(z_init.std()) * noise
    net.input_embedder.register_forward_post_hook(nudge)
  # every trunk pass overwrites: what is left is the last pass (cloned to numpy at once,
  # before the confidence head's in-place `zij +=`)
  net.embeddings_and_pairformer.register_forward_post_hook(keep('trunk'))

  def conf_hook(_layer, _inp, out):
    for k in ('logits_pae', 'logits_pde', 'logits_plddt', 'logits_resolved', 'pae', 'atom_plddts'):
      if k in out:
        got['conf.' + k] = _np(out[k])
  net.confidence_head.register_forward_post_hook(conf_hook)
  if os.environ.get('HF3_TAPS'):
    # dev: the trunk's seams, last pass (every pass overwrites), in HF3 token order
    ep = net.embeddings_and_pairformer

    def pre(_layer, inp):
      got['tap.z_init'] = _np(inp[1])                     # pair_init_act
      got['tap.z_prev'] = _np(inp[2])                     # pair_act carried in
    ep.register_forward_pre_hook(pre)
    ep.template_embedder.register_forward_post_hook(keep('tap.template_out'))
    ep.msa_module.register_forward_post_hook(keep('tap.msa_out'))
    ep.pairformer_stack[0].register_forward_post_hook(keep('tap.pf0'))

  # (a) the MSA row draw: MsaModule.forward's paddle.randperm, the only randperm in the model
  orig_randperm = paddle.randperm

  def randperm(n, *a, **kw):
    out = orig_randperm(n, *a, **kw)
    rows.append(_np(out))
    return out
  paddle.randperm = randperm
  # (b) the confidence head's CentreRandomAugmentation: scipy Rotation.random (numpy global
  # RNG) then paddle.normal([B,1,3]); recorded around the stock function
  orig_aug = diffusion.CentreRandomAugmentation
  augs = []

  def aug(x, mask, s_trans=1, uniform=False):
    rec = {}
    orig_rot, orig_normal = diffusion.Rotation, paddle.normal

    class Rot:
      @staticmethod
      def random(*a, **kw):
        r = orig_rot.random(*a, **kw)
        rec.setdefault('R', []).append(r.as_matrix())
        return r

    def normal(*a, **kw):
      out = orig_normal(*a, **kw)
      rec['t'] = _np(out)
      return out
    diffusion.Rotation, paddle.normal = Rot, normal
    try:
      out = orig_aug(x, mask, s_trans=s_trans, uniform=uniform)
    finally:
      diffusion.Rotation, paddle.normal = orig_rot, orig_normal
    augs.append(rec)
    return out
  diffusion.CentreRandomAugmentation = aug
  try:
    _run(model, batch, precision)
  finally:
    paddle.randperm = orig_randperm
    diffusion.CentreRandomAugmentation = orig_aug

  depth = cfg.model.embeddings_and_pairformer.msa_module.msa_depth
  s_inputs, _, _ = got['input_embedder']
  s, zz = got['trunk']
  P = lambda x: x[perm]
  PP = lambda x: x[perm][:, perm]
  pae_logits = got['conf.logits_pae'][0]
  pde_logits = got['conf.logits_pde'][0]
  ch = cfg.model.heads.confidence_head
  breaks_pde = np.linspace(0., ch.stride_pde * ch.b_pde, ch.b_pde - 1)
  breaks_pae = np.linspace(0., ch.stride_pae * ch.b_pae, ch.b_pae - 1)
  out = {
      's_inputs': P(s_inputs[0]), 's': P(s[0]), 'z': PP(zz[0]),
      'denoised': got['denoised'],
      'pae_logits': PP(pae_logits), 'pde_logits': PP(pde_logits),
      'plddt_logits': got['conf.logits_plddt'][0],
      'resolved_logits': got['conf.logits_resolved'][0],
      # expected values, as HF3's own get_metrics computes them (pae Angstrom, plddt 0-100
      # per FLAT ATOM); pde through the same stock helper on its own breaks
      'pae': PP(confidence.compute_predicted_aligned_error(
          logits=pae_logits, breaks=breaks_pae)['predicted_aligned_error']),
      'pde': PP(confidence.predicted_prob_to_value(pde_logits, breaks_pde)),
      # 0-1, as every runner reports it (the harness scales to ours' 0-100)
      'plddt': confidence.compute_plddt(got['conf.logits_plddt'][0]) / 100.0,
      'hf3_token_perm': perm,
  }
  for k, v in got.items():
    if k.startswith('tap.'):
      vs = v if isinstance(v, list) else [v]
      for j, x in enumerate(vs):
        x = x[0]
        out['%s.%d' % (k, j)] = PP(x) if x.ndim == 3 else P(x)
  for i, r in enumerate(rows):
    out['msa_rows_pass%d' % i] = r[:depth]
  if augs:
    a = augs[-1]
    # x_aug = (x - mean) @ R + t (row-vector convention, R as scipy's as_matrix)
    out['conf_aug_rot'] = np.asarray(a['R'][0], np.float32)
    out['conf_aug_trans'] = np.asarray(a['t'], np.float32).reshape(3)
  np.savez(out_npz, **{k: np.asarray(v) for k, v in out.items()})


def trajectory(model_name, out_dir, in_npz, out_npz):
  """HF3's REAL sampler at its inference defaults (200 steps, gamma0 0.8 below 1400
  tokens), one sample, every random draw the harness's: x0 (stock multiplies by c_list[0]),
  and per step the augmentation rotation R (stock: x @ R, so passed AS GIVEN), translation t,
  and the noise eps (drawn every step, also where gamma = 0)."""
  import paddle
  from helixfold.model import diffusion
  z = np.load(in_npz)
  model, batch, meta, cfg = _setup(out_dir, int(z['recycling_steps']))
  net = model.helixfold
  dm = net.diffusion_module
  x0 = paddle.to_tensor(z['traj_x0'], dtype='float32')
  eps = z['traj_eps'].astype(np.float32)
  rot = z['traj_rot'].astype(np.float64)
  trans = z['traj_trans'].astype(np.float32)
  n = {'normal_atom': 0, 'normal_t': 0, 'rot': 0}
  got = {}
  orig_sample = dm.sample_diffusion
  orig_sched = dm._noise_schedule

  def sched(step_num):
    c = orig_sched(step_num)
    got['sigmas'] = _np(c)
    return c
  dm._noise_schedule = sched

  class Rot:
    @staticmethod
    def random(*a, **kw):
      k = n['rot']
      n['rot'] += 1

      class _R:
        def as_matrix(self):
          return rot[k]
      return _R()

  def normal(mean=0.0, std=1.0, shape=None, name=None):
    shape = list(shape)
    if shape[-2:] == [1, 3]:
      k = n['normal_t']
      n['normal_t'] += 1
      return paddle.to_tensor(trans[k].reshape(shape), dtype='float32')
    k = n['normal_atom']
    n['normal_atom'] += 1
    v = x0 if k == 0 else paddle.to_tensor(eps[k - 1], dtype='float32')
    return v.reshape(shape)

  def sample_diffusion(representations, batch, step_num=None, gamma0=None):
    nudge = float(os.environ.get('TRAJ_TRUNK_NUDGE', 0) or 0)
    if nudge:
      # the harness's trunk floor: trunk outputs scaled by (1 + nudge * N(0,1)), a fixed draw
      g = np.random.default_rng(7)
      representations = dict(representations)
      for k in ('single', 'pair'):
        v = representations[k]
        representations[k] = v * paddle.to_tensor(
            1 + nudge * g.normal(size=v.shape).astype(np.float32))
    orig_rot, orig_normal = diffusion.Rotation, paddle.normal
    diffusion.Rotation, paddle.normal = Rot, normal
    try:
      ret = orig_sample(representations, batch, step_num=step_num, gamma0=gamma0)
    finally:
      diffusion.Rotation, paddle.normal = orig_rot, orig_normal
    got.setdefault('final', _np(ret['final_atom_positions']).reshape(-1, 3))
    return ret
  dm.sample_diffusion = sample_diffusion
  _run(model, batch, 'fp32')
  assert n['rot'] == n['normal_t'] == n['normal_atom'] - 1, n
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['rot'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
