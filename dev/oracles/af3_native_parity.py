"""alphafold3 against UPSTREAM DeepMind AlphaFold 3, every input modality.

Until this existed alphafold3 was N/A at every module level: it "is the
reference". It is not -- it is a fork, and 435b8dc found it running the atom
transformer's pair conditioning on the wrong matrix with nothing failing. This
compares it against the upstream release it forked from, on the same weights
(~/ported/alphafold3/af3.bin.zst is byte-identical to the official download).

  ~/venv/bin/python dev/oracles/af3_native_parity.py [CASE ...] [--bf16 | --bf16-sampler]

  --bf16          ours bfloat16='intermediate' against upstream's bf16: the SAME
                  precision split (trunk + confidence bf16, sampler fp32), so
                  this is parity.
  --bf16-sampler  ours bfloat16='all', the shipped default on Ampere and newer
                  (config_factory.bfloat16_default): the sampler in bf16 too,
                  which upstream never does. Measures that trade (301cc16)
                  against upstream's own bf16 error; not a port defect if it
                  fails, but a number to know.

Per case, three questions, each graded rather than printed:

  feat     Does OUR featurisation build the batch UPSTREAM's builds, from the
           same JSON? (integer fields exact; floats graded)
  model    On OUR batch, does our model compute what upstream's does?
           trunk single/pair after recycling, distogram, one denoise step on
           injected (positions_noisy, sigma), every confidence output on
           injected coordinates.
  cached   Does the shipped once-per-fold diffusion conditioning give the same
           denoise as building it inline? Only our side has both paths.

Both models run their REAL `Model.__call__`, so modules are created in the order
and scope a fold creates them. See af3_upstream_runner.py for why that matters.

Prints `__VERDICT PASS` or `__VERDICT FAIL` and exits non-zero on FAIL.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser('~')
UPSTREAM_PY = os.environ.get('AF3_UPSTREAM_PY', HOME + '/venv_af3/bin/python')
MODEL_DIR = os.environ.get('AF3_MODEL_DIR', HOME + '/ported/alphafold3')

UBQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
# 6MRR, the de-novo protein every other gate folds.
P6MRR = 'GSWKELAAKLAKEAAEKGDPEVAKRALELAKKAAEEGDPELAKKAAELAKKA'
DNA_F, DNA_R = 'GCGATCGCAT', 'ATGCGATCGC'


def _prot(cid, seq, msa=None, templates=None, mods=None):
  return {'protein': {
      'id': cid, 'sequence': seq, 'modifications': mods or [],
      'unpairedMsa': msa if msa is not None else '>q\n%s\n' % seq,
      'pairedMsa': '', 'templates': templates or []}}


def _template_5k9p():
  # Ubiquitin's own crystal structure as its template: a template case that
  # reads real coordinates, so the template embedder is not fed zeros.
  with open(HOME + '/5K9P.cif') as f:
    cif = f.read()
  return [{'mmcif': cif, 'queryIndices': list(range(76)),
           'templateIndices': list(range(76))}]


def cases():
  """name -> AF3 JSON. Small on purpose: parity, not accuracy, so every
  modality fits in one fp32 fold on a 23 GB card."""
  c = {
      'protein': [_prot('A', P6MRR)],
      'msa': [_prot('A', UBQ, msa=json.load(open(HOME + '/5k9p_msa.json'))
                    ['sequences'][0]['protein']['unpairedMsa'])],
      'homodimer': [_prot(['A', 'B'], P6MRR)],
      'heterodimer': [_prot('A', P6MRR), _prot('B', UBQ)],
      'rna': [{'rna': {'id': 'A', 'sequence': 'GCGGAUUUAGCUCAGUUGGGAGAGC',
                       'modifications': [], 'unpairedMsa': ''}}],
      'rna_mod': [{'rna': {'id': 'A', 'sequence': 'GCGGAUUUAGCUCAGUUGGGAGAGC',
                           'modifications': [{'modificationType': 'PSU',
                                              'basePosition': 7}],
                           'unpairedMsa': ''}}],
      # 5-methyl-dC at position 2: boltz's unknown-DNA class (`DN`), which our
      # 31 restype classes do not carry.
      'dna_mod': [{'dna': {'id': 'A', 'sequence': DNA_F, 'modifications': [
                      {'modificationType': '5CM', 'basePosition': 2}]}},
                  {'dna': {'id': 'B', 'sequence': DNA_R, 'modifications': []}}],
      'dna': [{'dna': {'id': 'A', 'sequence': DNA_F, 'modifications': []}},
              {'dna': {'id': 'B', 'sequence': DNA_R, 'modifications': []}}],
      'protein_dna': [_prot('A', P6MRR),
                      {'dna': {'id': 'B', 'sequence': DNA_F, 'modifications': []}},
                      {'dna': {'id': 'C', 'sequence': DNA_R, 'modifications': []}}],
      'ligand_ccd': [_prot('A', P6MRR), {'ligand': {'id': 'B', 'ccdCodes': ['ATP']}}],
      'ligand_smiles': [_prot('A', P6MRR), {'ligand': {
          'id': 'B', 'smiles': 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'}}],
      'ion': [_prot('A', P6MRR), {'ligand': {'id': 'B', 'ccdCodes': ['MG']}}],
      'ptm': [_prot('A', UBQ, mods=[{'ptmType': 'SEP', 'ptmPosition': 20}])],
      # N-glycan: NAG on Asn ND2. Exercises bondedAtomPairs and the
      # polymer-ligand bond features, which no other case reaches.
      'covalent': [_prot('A', 'GSWKELAANLTKEAAEKG'),
                   {'ligand': {'id': 'B', 'ccdCodes': ['NAG']}}],
      'template': [_prot('A', UBQ, templates=_template_5k9p())],
  }
  out = {}
  for name, seqs in c.items():
    j = {'name': name, 'modelSeeds': [1], 'sequences': seqs,
         'dialect': 'alphafold3', 'version': 3}
    if name == 'covalent':
      j['bondedAtomPairs'] = [[['A', 9, 'ND2'], ['B', 1, 'C1']]]
    out[name] = j
  return out


# ---------------------------------------------------------------- grading
# The thresholds parity_audit.py grades with, applied HERE, by the gate, so a
# number outside them is a non-zero exit and the driver cannot read it as OK.
def compare(a, b):
  a = np.asarray(a, np.float64).ravel()
  b = np.asarray(b, np.float64).ravel()
  d = np.abs(a - b)
  rms = float(np.sqrt((b ** 2).mean())) or 1.0
  if a.size <= 1:
    corr = 1.0
  elif a.std() == 0 or b.std() == 0:
    corr = 1.0 if np.array_equal(a, b) else 0.0
  else:
    corr = float(np.corrcoef(a, b)[0, 1])
  return corr, float(d.max()) / rms


# Half precision is graded as a DISTRIBUTION, across every case run, because
# one case cannot be: ours and upstream round the trunk differently, the
# confidence logits amplify whichever way it went, and a single case's ratio
# scatters x0.6-x2.2 with nothing wrong (measured over 14 cases, 2026-10-18:
# every array output averaged x0.91-x1.03). A real precision defect is
# SYSTEMATIC -- the bf16 sampler (301cc16) reads x1.6-x3.4 on every input.
#
#   per output, over cases: mean(ours_err / upstream_err) <= HALF_MEAN, and no
#   single case above HALF_MAX. Errors are relative rms against UPSTREAM FP32.
#   Scalars (average_pde) are a mean over a cancelling sum, so their upstream
#   error can be ~0 by chance; they are graded on |err| <= HALF_SCALAR instead.
HALF_MEAN, HALF_MAX, HALF_SCALAR = 1.25, 3.0, 0.05
HALF_SLACK, HALF_FLOOR = 1.5, 1e-4   # tap localisation only


HALF_STATS = {}   # output -> [(case, ours_err, upstream_err, size)]


def grade_half():
  """-> rows grading HALF_STATS as a distribution; see HALF_MEAN."""
  rows = []
  for k, st in sorted(HALF_STATS.items()):
    if st[0][3] <= 1:
      worst = max(st, key=lambda r: r[1])
      ok = worst[1] <= HALF_SCALAR
      rows.append(('half', '%s scalar: worst |err| %.2e (%s)'
                   % (k, worst[1], worst[0]), 'PARITY' if ok else 'BAD'))
      continue
    ratios = [(eo / eu if eu > HALF_FLOOR else (1.0 if eo <= HALF_FLOOR
                                                else float('inf')), c)
              for c, eo, eu, _ in st]
    mean = float(np.mean([r for r, _ in ratios]))
    mx, case = max(ratios)
    ok = mean <= HALF_MEAN and mx <= HALF_MAX
    rows.append(('half', '%s mean x%.2f  max x%.2f (%s)  over %d case(s)'
                 % (k, mean, mx, case, len(st)), 'PARITY' if ok else 'BAD'))
  return rows


def rel_rms_err(a, ref):
  a = np.asarray(a, np.float64)
  ref = np.asarray(ref, np.float64)
  return float(np.sqrt(((a - ref) ** 2).mean()) /
               (np.sqrt((ref ** 2).mean()) or 1.0))


def grade(corr, ratio):
  sys.path.insert(0, HERE)
  import parity_audit
  return parity_audit.grade(corr, ratio)


# ------------------------------------------------------------------ ours
def featurise_ours(j):
  from alphafold3.common import folding_input
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  fi = folding_input.Input.from_json(json.dumps(j))
  return featurisation.featurise_input(
      fold_input=fi, ccd=decoded_ccd.get_ccd(user_ccd=fi.user_ccd),
      buckets=None)[0]


def featurise_upstream(j, tmp):
  jp, op = os.path.join(tmp, 'in.json'), os.path.join(tmp, 'feat.npz')
  with open(jp, 'w') as f:
    json.dump(j, f)
  _upstream('featurise', jp, op)
  return dict(np.load(op))


def _upstream(*args):
  r = subprocess.run([UPSTREAM_PY, '-I', os.path.join(HERE,
                      'af3_upstream_runner.py'), *args],
                     capture_output=True, text=True, cwd=tempfile.gettempdir())
  if r.returncode:
    raise RuntimeError('upstream %s failed:\n%s' % (args[0], r.stderr[-4000:]))


def forward_ours(batch, pos, pos_noisy, sigma, bf16, num_recycles, seed):
  import haiku as hk
  import jax
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import model_registry
  from alphafold3.model import params as afp
  from alphafold3.model.components import utils

  cfg = af3_model.Model.Config()
  model_registry.get('alphafold3').configure(cfg)
  cfg.global_config.bfloat16 = bf16
  cfg.global_config.flash_attention_implementation = 'xla'
  cfg.num_recycles = num_recycles
  cfg.return_embeddings = True
  cfg.return_distogram = True
  pos, pos_noisy, sigma = map(jnp.asarray, (pos, pos_noisy, sigma))

  class Injected(af3_model.Model):

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config,
                          diffusion_state=None):
      # The shipped order: conditioning once, then the step that consumes it.
      pair_cond, atom_cond = self.diffusion_module(
          positions_noisy=None, noise_level=jnp.zeros(()), batch=batch,
          embeddings=embeddings, use_conditioning=True, conditioning_only=True)
      cached = self.diffusion_module(
          positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
          embeddings=embeddings, use_conditioning=True,
          pair_cond=pair_cond, atom_cond=atom_cond)
      # Tapped: this call builds its conditioning inline, as upstream's does,
      # so the two sides' module call sequences line up.
      with tapping(taps, outer):
        inline = self.diffusion_module(
            positions_noisy=pos_noisy, noise_level=sigma, batch=batch,
            embeddings=embeddings, use_conditioning=True)
      mask = batch.predicted_structure_info.atom_mask
      return {'atom_positions': pos[None], 'mask': mask[None],
              'denoised': cached[None], 'denoised_inline': inline[None]}

  sys.path.insert(0, HERE)
  from af3_upstream_runner import _flatten, tapping
  taps, outer = {}, [None]

  @hk.transform
  def fwd(b):
    outer[0] = jax.tree_util.tree_leaves(b)[0]._trace
    out = Injected(cfg)(b)
    return {**out, **taps}

  b = jax.tree_util.tree_map(jnp.asarray,
                             utils.remove_invalidly_typed_feats(batch))
  out = jax.jit(fwd.apply)(afp.get_model_haiku_params(model_dir=MODEL_DIR),
                           jax.random.PRNGKey(seed), b)
  return _flatten(jax.device_get(out))


# ------------------------------------------------------------------- run
# Fields whose values legitimately differ between two featurisations of the
# same input. Conformers are RDKit embeddings; both pipelines seed them, but
# the reference positions are graded rather than required equal.
FEAT_GRADED = ('ref_pos',)


def run_case(name, j, bf16, num_recycles=1, seed=1):
  from alphafold3.model.components import utils
  rows = []
  with tempfile.TemporaryDirectory() as tmp:
    ours_b = featurise_ours(j)
    ups_b = featurise_upstream(j, tmp)

    # --- feat
    ob = {k: np.asarray(v) for k, v in
          utils.remove_invalidly_typed_feats(ours_b).items()}
    only_o = sorted(set(ob) - set(ups_b))
    only_u = sorted(set(ups_b) - set(ob))
    if only_u:
      rows.append(('feat', 'missing in ours: %s' % ','.join(only_u), 'BAD'))
    for k in sorted(set(ob) & set(ups_b)):
      a, u = ob[k], ups_b[k]
      if a.shape != u.shape:
        rows.append(('feat', '%s shape %s vs %s' % (k, a.shape, u.shape), 'BAD'))
      elif a.dtype.kind in 'iub' or k not in FEAT_GRADED:
        if not np.array_equal(a, u):
          n = int((a != u).sum())
          rows.append(('feat', '%s %d/%d differ' % (k, n, a.size), 'BAD'))
      else:
        c, r = compare(a, u)
        rows.append(('feat', '%s corr %.6f max|d|/rms %.2e' % (k, c, r),
                     grade(c, r)))
    if only_o:
      rows.append(('feat', 'extra in ours (ignored by upstream): %s'
                   % ','.join(only_o), 'INFO'))

    # --- injected inputs, deterministic per case
    # AF3_INJECT_SEED varies the injected coordinates: one draw cannot tell a
    # systematic half-precision excess from where the rounding happened to fall.
    rng = np.random.default_rng(int(os.environ.get('AF3_INJECT_SEED', 0)))
    atom_mask = ob['pred_dense_atom_mask'].astype(np.float32)[..., None]
    pos = (rng.normal(size=atom_mask.shape[:2] + (3,)) * 8 * atom_mask
           ).astype(np.float32)
    sigma = np.float32(10.0)
    pos_noisy = (pos + rng.normal(size=pos.shape) * sigma).astype(np.float32)

    inp = os.path.join(tmp, 'fwd.npz')
    # Upstream has only on/off: its sampler is always fp32.
    np.savez(inp, positions=pos, positions_noisy=pos_noisy, noise_level=sigma,
             bfloat16='none' if bf16 == 'none' else 'all', num_recycles=num_recycles, seed=seed,
             model_dir=MODEL_DIR, **{'batch/' + k: v for k, v in ob.items()})
    _upstream('forward', inp, os.path.join(tmp, 'up.npz'))
    up = dict(np.load(os.path.join(tmp, 'up.npz')))
    # In half precision the reference is upstream at FP32, and the question is
    # whether ours lands as close to it as upstream's own half-precision run
    # does. Two graphs round bf16 in different places, so ours-vs-upstream at
    # bf16 measures both roundings at once and has no threshold of its own.
    up32 = None
    if bf16 != 'none':
      inp32 = os.path.join(tmp, 'fwd32.npz')
      z = dict(np.load(inp))
      z['bfloat16'] = 'none'
      np.savez(inp32, **z)
      _upstream('forward', inp32, os.path.join(tmp, 'up32.npz'))
      up32 = dict(np.load(os.path.join(tmp, 'up32.npz')))
    ours = forward_ours(ours_b, pos, pos_noisy, sigma, bf16, num_recycles, seed)
    # AF3_KEEP=dir keeps both sides' outputs, for localising by hand.
    if os.environ.get('AF3_KEEP'):
      keep = os.path.join(os.environ['AF3_KEEP'], name)
      os.makedirs(keep, exist_ok=True)
      np.savez(os.path.join(keep, 'upstream.npz'), **up)
      np.savez(os.path.join(keep, 'ours.npz'), **ours)
      np.savez(os.path.join(keep, 'batch_ours.npz'), **ob)
      np.savez(os.path.join(keep, 'batch_upstream.npz'), **ups_b)

  # --- taps: the first module, in upstream's call order, that disagrees.
  # Not graded on their own -- the outputs below are -- but they say WHERE.
  utaps = sorted((k for k in up if k.startswith('tap/')),
                 key=lambda k: int(up['taporder/' + k[4:]]))
  first = None
  for k in utaps:
    if k in ours and ours[k].shape == up[k].shape:
      c, r = compare(ours[k], up[k])
      if grade(c, r) != 'PARITY':
        first = (k, c, r)
        break
  # In half precision, the module whose output first falls behind upstream's
  # own half-precision error against upstream FP32 -- the same yardstick as the
  # outputs below, applied module by module.
  if up32 is not None:
    worst = []
    for k in utaps:
      if k in ours and k in up32 and ours[k].shape == up32[k].shape:
        eo, eu = rel_rms_err(ours[k], up32[k]), rel_rms_err(up[k], up32[k])
        worst.append((eo / (eu + 1e-6), k, eo, eu))
    first = next((w for w in worst if w[0] > HALF_SLACK and w[2] > HALF_FLOOR),
                 None)
    if first:
      rows.append(('tap', 'first half-precision excess %s ours %.2e upstream '
                   '%.2e (x%.2f)' % (first[1], first[2], first[3], first[0]),
                   'INFO'))
    for w in sorted(worst, reverse=True)[:5]:
      rows.append(('tap', 'excess x%.2f %s' % (w[0], w[1]), 'INFO'))
    first = None
  only_up = [k for k in utaps if k not in ours]
  if first:
    rows.append(('tap', 'first divergence %s corr %.6f max|d|/rms %.2e'
                 % first, 'INFO'))
  if only_up:
    rows.append(('tap', '%d upstream module calls absent in ours, e.g. %s'
                 % (len(only_up), ', '.join(only_up[:3])), 'INFO'))
  up = {k: v for k, v in up.items() if not k.startswith('tap')}
  ours = {k: v for k, v in ours.items() if not k.startswith('tap')}

  # --- model
  for k in sorted(set(up) & set(ours)):
    if up[k].shape != ours[k].shape:
      rows.append(('model', '%s shape %s vs %s' % (k, ours[k].shape,
                                                   up[k].shape), 'BAD'))
      continue
    if up[k].dtype.kind != 'f':
      ok = np.array_equal(up[k], ours[k])
      rows.append(('model', '%s %s' % (k, 'equal' if ok else 'DIFFER'),
                   'PARITY' if ok else 'BAD'))
      continue
    if up32 is not None and k in up32:
      eo, eu = rel_rms_err(ours[k], up32[k]), rel_rms_err(up[k], up32[k])
      HALF_STATS.setdefault(k, []).append((name, eo, eu, up32[k].size))
      rows.append(('model', '%s err-vs-fp32 ours %.2e upstream %.2e (x%.2f)'
                   % (k, eo, eu, eo / eu if eu else float('inf')), 'INFO'))
      continue
    c, r = compare(ours[k], up[k])
    rows.append(('model', '%s corr %.6f max|d|/rms %.2e' % (k, c, r),
                 grade(c, r)))
  missing = sorted(set(up) - set(ours))
  if missing:
    rows.append(('model', 'upstream outputs we lack: %s' % ','.join(missing),
                 'BAD'))

  # --- cached vs inline conditioning (ours only)
  c, r = compare(ours['diffusion_samples/denoised'],
                 ours['diffusion_samples/denoised_inline'])
  rows.append(('cached', 'denoise cached vs inline corr %.6f max|d|/rms %.2e'
               % (c, r), grade(c, r)))
  return rows


# bf16 rows are graded against upstream's fp32 (see HALF_SLACK), so both modes
# demand PARITY of what they grade.
PASSING = {'none': ('PARITY', 'INFO'),
           'intermediate': ('PARITY', 'INFO'),
           'all': ('PARITY', 'INFO')}


def check_env():
  """Refuse to compare across RDKit versions.

  Reference conformers are RDKit ETKDG embeddings, and two RDKit releases embed
  the same molecule from the same seed differently. Upstream bumped its pin to
  2026.3.3 after we forked at 2025.9.4, and that alone read as ref_pos corr 0.91
  -- a featurisation "bug" that was a library version. ~/venv_af3 must carry
  OUR pin: `uv pip install --python ~/venv_af3/bin/python rdkit==<ours>`.
  """
  import rdkit
  r = subprocess.run([UPSTREAM_PY, '-I', '-c',
                      'import rdkit; print(rdkit.__version__)'],
                     capture_output=True, text=True)
  theirs = r.stdout.strip()
  if theirs != rdkit.__version__:
    raise SystemExit('RDKit differs: ours %s, upstream venv %s. Align them; '
                     'see check_env.' % (rdkit.__version__, theirs))
  up = subprocess.run(['git', '-C', HOME + '/af3_upstream', 'describe',
                       '--always', '--dirty'], capture_output=True, text=True)
  print('upstream af3 %s  rdkit %s  weights %s'
        % (up.stdout.strip(), theirs, MODEL_DIR))


def main(argv):
  check_env()
  bf16 = ('all' if '--bf16-sampler' in argv else
          'intermediate' if '--bf16' in argv else 'none')
  names = [a for a in argv if not a.startswith('--')]
  allc = cases()
  names = names or list(allc)
  failed = []
  for n in names:
    print('== %s  (bfloat16=%s)' % (n, bf16), flush=True)
    try:
      rows = run_case(n, allc[n], bf16)
    except Exception as e:  # a crash is a FAILURE, never a skip
      import traceback
      traceback.print_exc()
      print('  ERROR %s: %s' % (type(e).__name__, str(e).splitlines()[0][:200]))
      failed.append(n)
      continue
    bad = [r for r in rows if r[2] not in PASSING[bf16]]
    for stage, msg, g in rows:
      print('  %-6s %-7s %s' % (stage, g, msg))
    print('  %s: %s' % (n, 'PASS' if not bad else 'FAIL (%d rows)' % len(bad)),
          flush=True)
    if bad:
      failed.append(n)
  if bf16 != 'none':
    print('== half precision, graded over %d case(s)' % len(names))
    hrows = grade_half()
    for stage, msg, g in hrows:
      print('  %-6s %-7s %s' % (stage, g, msg))
    if any(g != 'PARITY' for _, _, g in hrows):
      failed.append('half-precision')
  print('__VERDICT %s %s' % ('FAIL' if failed else 'PASS', ' '.join(failed)))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
  # tokamax parses sys.argv lazily, as absl flags, and dies on `--bf16`.
  argv, sys.argv = sys.argv[1:], sys.argv[:1]
  sys.exit(main(argv))
