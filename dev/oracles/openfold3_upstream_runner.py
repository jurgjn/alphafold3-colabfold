"""The STOCK OpenFold3 half of vendor_native_parity.py (openfold3, openbind0).

  python openfold3_upstream_runner.py featurise MODEL CASE.json OUT_DIR
  python openfold3_upstream_runner.py forward   MODEL OUT_DIR IN.npz OUT.npz

ONE runner, two releases (openfold3: tag 0.4.5 in ~/openfold-3-045; openbind0:
tag v0.5.0 in ~/openfold-3-v050): the MODEL picks the tree (VENDORS in
vendor_native_parity.py puts exactly that tree on PYTHONPATH; this file checks
it got the right one) and the checkpoint. openbind0 is OF3 v0.5.0 and the
RELEASE matters: v0.5.0's end-node triangle attention transposes the pair bias
and main's does not, while main loads openbind0's tensors without complaint
(run_all_parity.sh). Both run in ~/venv_of3 (the two trees' conda locks are
identical; see ~/venv_of3/requirements-of3.txt).

Everything is OpenFold3's own: the run is the vendor's `run_openfold.py
predict` CLI itself (click, `standalone_mode=False`), so the config is
InferenceExperimentConfig + the `predict` preset, the batch comes from
InferenceDataModule's DataLoader (worker-seeded from data_seed, as stock), the
checkpoint is loaded by InferenceExperimentRunner.setup (EMA weights, strict),
and the forward is Lightning's predict loop -> OpenFold3AllAtom.predict_step
(reseed, model forward, confidence scores). What the runner yaml we hand that
CLI sets is listed in _runner_yaml, each with its reason. The single model
override is SampleDiffusion.forward -- the only stochastic step -- which runs
the real DiffusionModule ONCE on the injected (noised coordinates, sigma) and
returns the injected coordinates, which the confidence heads then score. See
boltz2_upstream_runner.py for why.

ORDER is the CLI's: model built and loaded (setup), then the dataloader
featurises (ref_pos's random augmentation -- centre_random_augmentation in
featurization/conformer.py, translation at scale 1.0 -- is drawn in a DataLoader
worker seeded from data_seed), then predict_step reseeds and predicts, one
process. `featurise` runs the SAME CLI and stops at predict_step, so its saved
draw is the one `forward` sees; `forward` checks that and fails if not.
"""
import json
import os
import sys

import numpy as np

H = os.path.expanduser('~')
# model -> (vendor tree, checkpoint). converters/sources.py names the files.
MODELS = {
    # STOCK RELEASE 0.4.5 (`git describe`: 0.4.5, a4b08032; worktree of
    # ~/openfold-3 at the tag). It is the newest release that loads preview-2:
    # OF3's own registry (entry_points/parameters.py) marks of3-p2-155k
    # ">=0.4,<0.4.4dev0" on origin/main, and origin/main (v0.5.0-155-g8090aa69,
    # MODEL_VERSION 2.0.0) refuses it on strict load -- the diffusion
    # transformer's layer_norm_z moved out of attention_pair_bias (measured
    # 2026-10-19). 0.4.5 has MODEL_VERSION 1.0.0 and lists p2-155k ">=0.4".
    # The old ~/openfold-3 checkout (c615a7f8) is 0.4.3 + 1 commit.
    'openfold3': (H + '/openfold-3-045', os.environ.get(
        'OF3_CKPT', H + '/of3-p2-155k.pt')),
    # of3-ob-2025-06-30-174k.pt in converters/sources.py, saved here as
    # of3-ob-174k.pt (same 2287872989-byte file the other openbind0 gates use).
    'openbind0': (H + '/openfold-3-v050', os.environ.get(
        'OB0_CKPT', H + '/of3-ob-174k.pt')),
}
SEED = 42   # the CLI's default model seed (InferenceExperimentSettings.seeds)


def _diag_fix_deletion():
  """DIAGNOSTIC ONLY (OF3_DIAG_FIX_DELETION=1), never the gate of record.

  Release 0.4.5 scales the MSA deletion value by `2.0 / torch.acos(zeros(1)) * 2` -- 2 / (pi/2) * 2 = 8/pi,
  four times AF3's 2/pi (operator precedence; v0.5.0 writes `2.0 / torch.pi`). Every MSA with an insertion
  reads deletion values x4 (deep_msa: raw 3 -> 2.0 where AF3 has 0.5). This rescales 0.4.5's output to 2/pi
  so the rest of its MSA path can be compared; a no-op on a tree without the bug."""
  import inspect
  import openfold3.core.data.pipelines.featurization.msa as fm
  if 'acos' not in inspect.getsource(fm.MsaFeaturizerOF3.create_features):
    return
  orig = fm.MsaFeaturizerOF3.create_features

  def create_features(self, *a, **kw):
    out = orig(self, *a, **kw)
    if 'deletion_value' in out:
      out['deletion_value'] = out['deletion_value'] / 4.0
    return out
  fm.MsaFeaturizerOF3.create_features = create_features


def _diag_fix_paired():
  """DIAGNOSTIC ONLY (OF3_DIAG_FIX_PAIRED=1), never the gate of record.

  Release 0.4.5's create_paired_from_precomputed builds each chain's paired block but never records its row count
  (set_row_counts(n_rows_paired_subsampled=...), added in v0.5.0), so the stack takes ZERO paired rows: a precomputed
  paired MSA is silently dropped, while the main MSA is still deduplicated against it (paired_msa: 49 rows, the
  query plus main-minus-paired, where v0.5.0 has 74). This records the count as v0.5.0 does; a no-op on a tree
  that already does."""
  import inspect
  import openfold3.core.data.pipelines.sample_processing.msa as sm
  if 'set_row_counts' in inspect.getsource(sm.create_paired_from_precomputed):
    return
  orig = sm.create_paired_from_precomputed

  def create_paired_from_precomputed(msa_array_collection, *a, **kw):
    out = orig(msa_array_collection, *a, **kw)
    if out:
      msa_array_collection.set_row_counts(
          n_rows_paired_subsampled=int(next(iter(out.values())).msa.shape[0]))
    return out
  sm.create_paired_from_precomputed = create_paired_from_precomputed


def _diag_fix_profile():
  """DIAGNOSTIC ONLY (OF3_DIAG_FIX_PROFILE=1), never the gate of record.

  Release 0.4.5 (like every 0.4.x; fixed on main by f896db32, 2026-03-19,
  shipped in v0.5.0 only) computes the MSA profile with
  `col_indices_local = np.repeat(np.arange(block_n_cols), n_rows)` against a
  row-major `msa_chunk.ravel()` (primitives/sequence/msa.py:calculate_profile),
  so profile[c] counts row-major cells c*n_rows..c*n_rows+n_rows-1 instead of
  column c; exact only for a 1-row MSA. v0.5.0 has `np.tile` (correct). This
  swaps in the v0.5.0 line so the REST of the MSA path can be compared. It runs
  at import, so DataLoader workers (forkserver re-imports __main__ as
  __mp_main__) are patched too.
  """
  import openfold3.core.data.pipelines.sample_processing.msa as sp
  import openfold3.core.data.primitives.sequence.msa as sq
  from openfold3.core.data.primitives.sequence.msa import (
      STANDARD_RESIDUES_WITH_GAP_1, map_str_array_to_idx_array)

  def calculate_profile(msa_array, molecule_type, chunk_size):
    idx = map_str_array_to_idx_array(msa_array, molecule_type)
    n_rows, n_cols = idx.shape
    k = len(STANDARD_RESIDUES_WITH_GAP_1)
    flat = np.tile(np.arange(n_cols), n_rows) * k + idx.ravel()
    return np.bincount(flat, minlength=n_cols * k).reshape(n_cols, k) / n_rows
  sq.calculate_profile = sp.calculate_profile = calculate_profile


if os.environ.get('OF3_DIAG_FIX_PROFILE'):
  _diag_fix_profile()
if os.environ.get('OF3_DIAG_FIX_DELETION'):
  _diag_fix_deletion()
if os.environ.get('OF3_DIAG_FIX_PAIRED'):
  _diag_fix_paired()


def _check_tree(model):
  import openfold3
  want = os.path.realpath(MODELS[model][0])
  got = os.path.realpath(os.path.dirname(os.path.dirname(openfold3.__file__)))
  if got != want:
    # A wrong overlay is a silent wrong answer, not an ImportError.
    raise RuntimeError('%s needs %s, imported openfold3 from %s'
                       % (model, want, got))


def _trivial_msa(msa, seq):
  return (not msa) or msa.strip() == '>q\n%s' % seq


def _of3_query(j, msa_root):
  """AF3 fold-input JSON -> OF3's InferenceQuerySet JSON.

  docs/source/input_format_reference.md. A chain with an alignment gets it as a
  precomputed MSA file named `colabfold_main.a3m` (`colabfold_paired.a3m`),
  the names OF3's own MSA-server path writes and its MSASettings know
  (aln_order / max_seq_counts); a chain with none gets NO path, so
  InferenceDataModule.prepare_data gives it OF3's single-sequence dummy MSA
  exactly as the CLI does with --use_msa_server false.
  """
  chains = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    if kind in ('protein', 'rna', 'dna'):
      c = {'molecule_type': kind, 'chain_ids': ids, 'sequence': e['sequence']}
      # 1-based residue index -> CCD code.
      mods = {str(m['ptmPosition']): m['ptmType']
              for m in e.get('modifications', []) if 'ptmType' in m}
      mods.update({str(m['basePosition']): m['modificationType']
                   for m in e.get('modifications', [])
                   if 'modificationType' in m})
      if mods:
        c['non_canonical_residues'] = mods
      for key, fname, field in (
          ('unpairedMsa', 'colabfold_main.a3m', 'main_msa_file_paths'),
          ('pairedMsa', 'colabfold_paired.a3m', 'paired_msa_file_paths')):
        if not _trivial_msa(e.get(key), e['sequence']):
          d = os.path.join(msa_root, ids[0])
          os.makedirs(d, exist_ok=True)
          with open(os.path.join(d, fname), 'w') as f:
            f.write(e[key])
          c[field] = [os.path.join(d, fname)]
      if e.get('templates'):
        # OF3's query format takes template CIFs per chain (template_cif_paths,
        # template_cif_chain_ids) and aligns them itself
        d = os.path.join(msa_root, ids[0])
        os.makedirs(d, exist_ok=True)
        paths, chains = [], []
        for k, t in enumerate(e['templates']):
          p = os.path.join(d, 'template_%d.cif' % k)
          with open(p, 'w') as f:
            f.write(t['mmcif'])
          import gemmi
          paths.append(p)
          chains.append(gemmi.cif.read(p).sole_block().find_value('_entity_poly.pdbx_strand_id').split(',')[0].strip())
        c['template_cif_paths'] = paths
        c['template_cif_chain_ids'] = chains
    elif kind == 'ligand':
      # An AF3 ligand, MG included, is a CCD (or SMILES) ligand chain; OF3's
      # query format has no separate ion type.
      c = {'molecule_type': 'ligand', 'chain_ids': ids}
      if 'smiles' in e:
        c['smiles'] = e['smiles']
      else:
        c['ccd_codes'] = list(e['ccdCodes'])
    else:
      raise NotImplementedError(kind)
    chains.append(c)
  # bondedAtomPairs: NOT TRANSLATED, because stock OF3 cannot take them. Its
  # Query has a `covalent_bonds` field, but (a) Atom.atom_id is an int where
  # AF3 names the atom, and (b) nothing in openfold3/ reads covalent_bonds --
  # the inference featuriser builds token_bonds from the parsed structure only.
  # So the CLI runs the covalent case without the bond, and that is the vendor.
  # So the input is NOT EXPRESSIBLE in stock OF3: exit 3, which
  # vendor_native_parity reports as N/A (dropping the bond would compare a
  # different input -- NAG then keeps its leaving O1, 33 tokens to our 32).
  if j.get('bondedAtomPairs'):
    print('VENDOR_NA: stock OpenFold3 has no covalent-bond input '
          '(Query.covalent_bonds is never read)', file=sys.stderr)
    sys.exit(3)
  return {'seeds': [SEED], 'queries': {j['name']: {'chains': chains}}}


def _runner_yaml(model, out_dir, recycles):
  import torch
  custom = {
      'settings': {'memory': {'eval': {
          # Kernels are an implementation choice, not the model: torch
          # everywhere. The eval default turns Triton triangle attention ON
          # (model_config.py); deepspeed/cuequivariance are already off and
          # not installed in ~/venv_of3.
          'use_triton_triangle_kernels': False,
          'use_deepspeed_evo_attention': False,
          'use_cueq_triangle_kernels': False,
          'use_lma': False}}},
  }
  if recycles is not None:
    # The model reads shared.num_recycles at inference (model.py forward).
    custom['architecture'] = {'shared': {'num_recycles': int(recycles)}}
  if os.environ.get('MSA_ALL_ROWS'):
    # OF3 subsamples EXACTLY 1024 MSA rows at random each pass (msa_module_embedder subsample_all_msa,
    # min = max = 1024): a draw, which the harness lifts on every side -- all rows, as ours takes them
    custom.setdefault('architecture', {})['msa'] = {'msa_module_embedder': {
        'min_subsampled_all_msa': 1 << 20, 'max_subsampled_all_msa': 1 << 20}}
  return {
      # 'predict' is the CLI default preset (InferenceExperimentConfig);
      # passing model_update replaces the default, so it is restated.
      'model_update': {'presets': ['predict'], 'custom': custom},
      # fp32: the CLI default precision is already "32-true" (no bf16
      # autocast); restated so it cannot drift. Device: the vendor default
      # is gpu; CPU when the harness hides the GPU (VENDOR_CPU).
      'pl_trainer_args': {
          'precision': '32-true',
          'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
          'devices': 1},
      'output_writer_settings': {'write_full_confidence_scores': False},
  }


def _run_cli(model, out_dir, recycles, n_samples=1):
  """`run_openfold.py predict` exactly, with our yaml."""
  import yaml
  import torch
  rp = os.path.join(out_dir, 'runner.yml')
  with open(rp, 'w') as f:
    yaml.safe_dump(_runner_yaml(model, out_dir, recycles), f)
  # An isolated OPENFOLD_CACHE: v0.5.0's predict merges a user default
  # $OPENFOLD_CACHE/runner.yml under ours if one exists; none must leak in.
  os.environ['OPENFOLD_CACHE'] = os.path.join(out_dir, 'of3_cache')
  import openfold3.run_openfold as ro
  args = ['predict', '--query_json', os.path.join(out_dir, 'query.json'),
          '--inference_ckpt_path', MODELS[model][1],
          '--num_diffusion_samples', str(n_samples),
          '--runner_yaml', rp, '--use_msa_server', 'false',
          '--use_templates', 'true' if '"template_cif_paths"' in open(os.path.join(out_dir, 'query.json')).read()
          else 'false',
          '--output_dir', os.path.join(out_dir, 'of3_out')]
  # TF32: both releases' predict turn it on (torch.set_float32_matmul_precision
  # ("high")), so on a GPU every fp32 matmul rounds to a 10-bit mantissa while
  # ours runs at 'highest'. A precision setting, not the model: off for the
  # strict comparison. v0.5.0 has a flag for it; main calls _torch_gpu_setup
  # unconditionally, so 'highest' is restored right after it. OF3_TF32=1 keeps it.
  tf32 = bool(os.environ.get('OF3_TF32'))
  if hasattr(ro, '_enable_tf32'):
    args += ['--use_tf32', 'true' if tf32 else 'false']
  elif not tf32:
    orig = ro._torch_gpu_setup

    def _setup():
      orig()
      torch.set_float32_matmul_precision('highest')
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False
    ro._torch_gpu_setup = _setup
  ro.cli(args, standalone_mode=False)


def _layout(batch, out_dir):
  # Batch axis 0 (bs = 1); no padding at bs = 1 (the collator pads across a
  # batch only), but token_mask/atom_mask are carried through anyway.
  # On the host: on a GPU run Lightning has already moved the batch to cuda.
  b = {k: v[0].cpu() for k, v in batch.items() if hasattr(v, 'shape')}
  chars = b['ref_atom_name_chars'].argmax(-1).numpy()          # (atoms, 4)
  names = [''.join(chr(int(c) + 32) for c in row).strip() for row in chars]
  n_at = b['num_atoms_per_token'].numpy()
  start = b['start_atom_index'].numpy()
  atom_token = np.repeat(np.arange(len(n_at)), n_at)
  assert np.array_equal(start, np.concatenate([[0], np.cumsum(n_at)[:-1]]))
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names), atom_token=atom_token,
           atom_pad=b['atom_mask'].numpy() > 0,
           token_pad=b['token_mask'].numpy() > 0,
           # Every batch feature too, as boltz2's runner does, so a
           # featurisation disagreement can be localised from B2_KEEP alone.
           **{'feat.' + k: v.float().numpy().copy() if v.is_floating_point()
              else v.numpy().copy() for k, v in b.items()
              if k not in ('ref_pos', 'ref_space_uid', 'ref_element')},
           **{'feat.ref_pos': b['ref_pos'].float().numpy().copy(),
              'feat.ref_space_uid': b['ref_space_uid'].numpy().copy(),
              # OF3 one-hots element as atomic number - 1 (conformer.py);
              # the harness compares atomic numbers.
              'feat.ref_element': b['ref_element'].argmax(-1).numpy() + 1})


def _prep(model, out_dir):
  _check_tree(model)


def featurise(model, case_json, out_dir):
  _prep(model, out_dir)
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  with open(os.path.join(out_dir, 'query.json'), 'w') as f:
    json.dump(_of3_query(j, os.path.join(out_dir, 'msa')), f)
  from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

  def predict_step(self, batch, batch_idx):
    if not batch.get('valid_sample'):
      raise RuntimeError('openfold3 featurisation failed (see log above)')
    # CLONED (detach/clone, numpy .copy()): never a view of a live tensor.
    _layout({k: v.detach().clone() if hasattr(v, 'detach') else v
             for k, v in batch.items()}, out_dir)
    return None   # featurise only: the writer skips a None output
  OpenFold3AllAtom.predict_step = predict_step
  _run_cli(model, out_dir, recycles=None)
  if not os.path.exists(os.path.join(out_dir, 'layout.npz')):
    raise RuntimeError('predict_step never ran')


def forward(model, out_dir, in_npz, out_npz):
  import torch
  _prep(model, out_dir)
  z = np.load(in_npz)
  if str(z['precision']) != 'fp32':
    # the vendor's STOCK precision, for bf16 grading: OF3's predict is fp32
    # ("32-true") with TF32 matmuls on, which the strict comparison turns off
    os.environ['OF3_TF32'] = '1'
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  pos = torch.tensor(z['positions'], dtype=torch.float32)
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32)
  sigma = float(z['noise_level'])
  got, taps, calls = {}, {}, {}

  from openfold3.core.model.structure.diffusion_module import SampleDiffusion
  from openfold3.projects.of3_all_atom.model import OpenFold3
  from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

  # THE override: SampleDiffusion.forward -> one real DiffusionModule call on
  # the injected state, with the arguments the stock loop passes it, and the
  # injected coordinates back. Shapes as the stock loop's: xl is
  # [batch, samples, atoms, 3], t a 0-d tensor.
  def sample_forward(self, batch, si_input, si_trunk, zij_trunk,
                     noise_schedule, no_rollout_samples, use_conditioning=True,
                     **kw):
    dev = batch['atom_mask'].device
    dt = batch['atom_mask'].dtype
    got['denoised'] = self.diffusion_module(
        batch=batch, xl_noisy=noisy.to(dev, dt)[None, None],
        token_mask=batch['token_mask'], atom_mask=batch['atom_mask'],
        t=torch.tensor(sigma, device=dev, dtype=noise_schedule.dtype),
        si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
        use_conditioning=use_conditioning, **kw).detach().clone()
    return pos.to(dev, dt)[None, None].expand(
        batch['atom_mask'].shape[0], no_rollout_samples, -1, -1).clone()
  SampleDiffusion.forward = sample_forward

  orig_trunk = OpenFold3.run_trunk

  def run_trunk(self, batch, num_cycles, inplace_safe=False):
    s_input, s, zz = orig_trunk(self, batch, num_cycles, inplace_safe)
    # CLONED: OF3 runs inplace_safe=True at inference and adds into these
    # tensors afterwards; on CPU `.float().cpu().numpy()` would be a view.
    got['s_inputs'], got['s'], got['z'] = (
        s_input.detach().clone(), s.detach().clone(), zz.detach().clone())
    got['num_cycles'] = torch.tensor(num_cycles)
    return s_input, s, zz
  OpenFold3.run_trunk = run_trunk

  def hook(name):
    def f(_m, _i, o):
      n = calls[name] = calls.get(name, -1) + 1
      tag = name if n == 0 else '%s@%d' % (name, n)
      for k, x in enumerate(o if isinstance(o, (tuple, list)) else (o,)):
        if hasattr(x, 'float'):
          taps['tap.%s.%d' % (tag, k)] = x.detach().float().cpu().numpy().copy()
    return f

  orig_init = OpenFold3.__init__

  def init(self, *a, **kw):
    orig_init(self, *a, **kw)
    # Stage taps: every top-level submodule's outputs, plus the input
    # embedder's parts, for localisation.
    for name, m in self.named_children():
      m.register_forward_hook(hook(name))
    for name, m in self.input_embedder.named_children():
      m.register_forward_hook(hook('input_embedder.' + name))
    # OF3_DEEP_TAPS=1: every submodule of the MSA module and of pairformer
    # block 0 too (first call only, `tap.deep.<order>.<path>.k` in call
    # order), to place a trunk disagreement at one op.
    if os.environ.get('OF3_DEEP_TAPS'):
      seen = set()

      def deep(name):
        def f(_m, _i, o):
          if name in seen:
            return
          seen.add(name)
          for k, x in enumerate(o if isinstance(o, (tuple, list)) else (o,)):
            if hasattr(x, 'float'):
              taps['tap.deep.%04d.%s.%d' % (len(seen), name, k)] = (
                  x.detach().float().cpu().numpy().copy())
        return f
      for root, mod in (('msa_module', self.msa_module),
                        ('pairformer.0', self.pairformer_stack.blocks[0])):
        for name, m in mod.named_modules():
          if name:
            m.register_forward_hook(deep(root + '.' + name))
  OpenFold3.__init__ = init

  orig_step = OpenFold3AllAtom.predict_step

  def predict_step(self, batch, batch_idx):
    # The draw this run featurised must be the one the harness injected.
    rp = batch['ref_pos'][0].detach().float().cpu().numpy()
    if not np.array_equal(rp, lay['feat.ref_pos']):
      raise RuntimeError('ref_pos draw differs from featurise (max %.3g)'
                         % np.abs(rp - lay['feat.ref_pos']).max())
    out = orig_step(self, batch, batch_idx)
    if out is None:
      raise RuntimeError('openfold3 predict_step failed (see predict_err log)')
    _b, o = out
    got['distogram_logits'] = o['distogram_logits'].detach().clone()
    for k in ('pae_logits', 'pde_logits', 'plddt_logits'):
      got[k] = o[k].detach().clone()
    cs = o['confidence_scores']
    got['pae'] = cs['pae'].detach().clone()
    got['pde'] = cs['pde'].detach().clone()
    # OF3 reports pLDDT x100 (aggregate_confidence_ranking.py); the contract
    # is boltz's [0, 1] (the harness rescales to ours).
    got['plddt'] = cs['plddt'].detach().clone() / 100.0
    return out
  OpenFold3AllAtom.predict_step = predict_step

  _run_cli(model, out_dir, recycles=int(z['recycling_steps']))
  if 'denoised' not in got or 'plddt' not in got:
    raise RuntimeError('openfold3 forward incomplete: got %s' % sorted(got))
  if int(got['num_cycles']) != int(z['recycling_steps']) + 1:
    raise RuntimeError('ran %d cycles' % int(got['num_cycles']))
  out = {**got, **taps}
  np.savez(out_npz, **{k: v.float().cpu().numpy().squeeze()
                       if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in out.items()})


def trajectory(model, out_dir, in_npz, out_npz):
  """OF3's REAL rollout at `predict`'s defaults (200 steps), one sample, every
  random draw the harness's (vendor_native_parity --trajectory): the initial
  positions, and per step the augmentation (centre_random_augmentation) and
  the noise (torch.randn_like). OF3 rotates as x @ R^T; the replacement
  applies the harness's x @ R directly, keeping OF3's masked centring and
  mask multiply."""
  import torch
  _prep(model, out_dir)
  z = np.load(in_npz)
  from openfold3.core.model.structure import diffusion_module as D
  from openfold3.core.model.structure.diffusion_module import SampleDiffusion
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32)
  n = {'randn': 0, 'eps': 0, 'aug': 0}
  got = {}

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def randn(self, size, device=None, dtype=None, **kw):
      assert n['randn'] == 0, 'one randn (the initial positions) expected'
      n['randn'] += 1
      return x0.reshape(size).to(device=device, dtype=dtype or torch.float32)

    def randn_like(self, x, **kw):
      k = n['eps']
      n['eps'] += 1
      return eps[k].reshape(x.shape).to(device=x.device, dtype=x.dtype)

  def aug(xl, atom_mask, scale_trans=1.0):
    k = n['aug']
    n['aug'] += 1
    m = atom_mask[..., None]
    mean = (xl * m).sum(-2, keepdim=True) / m.sum(-2, keepdim=True)
    r, t = rot[k].to(xl), trans[k].to(xl)
    return ((xl - mean) @ r + scale_trans * t) * m

  D.torch, D.centre_random_augmentation = _Torch(), aug
  orig = SampleDiffusion.forward

  def forward_(self, batch, si_input, si_trunk, zij_trunk, noise_schedule,
               no_rollout_samples, *a, **kw):
    got['sigmas'] = noise_schedule.detach().float().cpu().numpy()
    out = orig(self, batch, si_input, si_trunk, zij_trunk, noise_schedule,
               no_rollout_samples, *a, **kw)
    got.setdefault('final', out.detach().float().cpu().numpy().reshape(-1, 3))
    return out
  SampleDiffusion.forward = forward_
  _run_cli(model, out_dir, recycles=int(z['recycling_steps']))
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
