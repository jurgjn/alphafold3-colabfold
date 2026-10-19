"""The STOCK OpenDDE half of vendor_native_parity.py (opendde).

  python opendde_upstream_runner.py featurise opendde CASE.json OUT_DIR
  python opendde_upstream_runner.py forward   opendde OUT_DIR IN.npz OUT.npz

Run with ONLY ~/OpenDDE (tag v1.1.0) on PYTHONPATH, in ~/venv_protenix (every
pin in OpenDDE's pyproject is satisfied there exactly; see VENDORS). Everything
is OpenDDE's own: the config built by `build_inference_config` from a CLI arg
string, exactly as `runner/inference.py:run` builds it; the batch from
`get_inference_dataloader`; the model and checkpoint through `InferenceRunner`;
the forward through `InferenceRunner.predict`. The single override is the
model's `sample_diffusion` -- the only stochastic step -- which returns the
injected coordinates and runs the real denoiser once on an injected (noised
coordinates, sigma). See boltz2_upstream_runner.py for why.

TOKEN SPACES. OpenDDE runs its trunk on residue tokens, then expands to
"structural tokens" (backbone / side-chain subtokens) for the diffusion module.
Its config has `structural_token_expansion.pair_output_space = 'residue'`, so
the distogram and the confidence head run on the RESIDUE trunk
(`select_pair_output_branch`), with the structural diffusion's atom
coordinates; the public PAE/PDE are therefore residue x residue and pLDDT is
per atom. The atom set and order are the same in both token spaces (only
atom_to_token_idx differs), so `positions`/`denoised` are in the vendor's flat
atom order and the layout below is the residue one.
"""
import json
import os
import sys

# OpenDDE's fused LayerNorm is a JIT-compiled CUDA kernel, chosen at import by
# this variable (opendde/model/layer_norm); an implementation choice, so the
# stock torch LayerNorm is the reference. ("torch" is also OpenDDE's default.)
os.environ.setdefault('LAYERNORM_TYPE', 'torch')

import numpy as np

CKPT = os.path.expanduser(os.environ.get('OPENDDE_CKPT',
                                         '~/opendde_weights/opendde.pt'))
NAMES = {'opendde': 'opendde_v1'}



def _vendor_na(msg):
  """The stock vendor cannot express this input: exit 3, graded N/A by the harness."""
  print('VENDOR_NA: ' + msg, file=sys.stderr)
  sys.exit(3)

def _trivial_msa(msa, seq):
  return (not msa) or msa.strip() == '>q\n%s' % seq



def _template_dir(out_dir):
  return os.path.join(out_dir, 'templates')


def _template_a3m(e, cid, out_dir):
  """The case's AF3 templates as OpenDDE's template search output (templatesPath, an hmmsearch a3m
  `>pdbid_chain/start-end mol:protein length:N`, the hit aligned to the query), each template's mmCIF pre-placed in
  an offline mmcif dir (<pdbid>.cif, fetch_remote off) and its release date in a release-dates file. OpenDDE's
  template code is protenix's: it re-aligns the query to the chain's full SEQRES with kalign and prefilters the hit
  (data/template/template_utils.py)."""
  import re
  from opendde.data.template.template_parser import TemplateParser
  tdir = _template_dir(out_dir)
  os.makedirs(os.path.join(tdir, 'mmcif'), exist_ok=True)
  dates_path = os.path.join(tdir, 'release_dates.json')
  dates = json.load(open(dates_path)) if os.path.exists(dates_path) else {}
  q = e['sequence']
  lines = []
  for t in e['templates']:
    cif = t['mmcif']
    pdb_id = re.search(r'_entry\.id\s+(\S+)', cif).group(1).lower()
    with open(os.path.join(tdir, 'mmcif', '%s.cif' % pdb_id), 'w') as f:
      f.write(cif)
    chain = re.search(r'_entity_poly\.pdbx_strand_id\s+(\S+)', cif)
    chain = (chain.group(1).split(',')[0] if chain else 'A').strip("'\"")
    res = TemplateParser.parse(file_id=pdb_id, mmcif_string=cif, auth_chain_id=chain)
    seqres = res.mmcif_object.chain_to_seqres[chain]
    dates[pdb_id] = {'release_date': res.mmcif_object.header['release_date']}
    qmap = dict(zip(t['queryIndices'], t['templateIndices']))
    aligned = ''.join(seqres[qmap[i]] if i in qmap else '-' for i in range(len(q)))
    ts = sorted(qmap.values())
    lines += ['>%s_%s/%d-%d [subseq from] mol:protein length:%d  case template'
              % (pdb_id, chain, ts[0] + 1, ts[-1] + 1, len(seqres)), aligned]
  with open(dates_path, 'w') as f:
    json.dump(dates, f)
  path = os.path.join(tdir, '%s.a3m' % cid)
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  return path

def _opendde_json(j, out_dir=None):
  """AF3 fold-input JSON -> OpenDDE's input JSON (a one-element job list).

  docs/infer_json_format.md. MSAs go INLINE (`unpairedMsa`/`pairedMsa`, which
  InferenceMSAFeaturizer reads before the *Path fields), exactly as the AF3
  JSON carries them; a query-only MSA is treated as none, as for protenix.
  """
  seqs, entity_of = [], {}
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    for cid in ids:
      entity_of[cid] = len(seqs) + 1
    if kind == 'protein':
      d = {'sequence': e['sequence'], 'count': len(ids)}
      mods = [{'ptmType': 'CCD_' + m['ptmType'], 'ptmPosition': m['ptmPosition']}
              for m in e.get('modifications', [])]
      if mods:
        d['modifications'] = mods
      if not _trivial_msa(e.get('unpairedMsa'), e['sequence']):
        d['unpairedMsa'] = e['unpairedMsa']
      if not _trivial_msa(e.get('pairedMsa'), e['sequence']):
        d['pairedMsa'] = e['pairedMsa']
      if e.get('templates'):
        d['templatesPath'] = _template_a3m(e, ids[0], out_dir)
      seqs.append({'proteinChain': d})
    elif kind in ('rna', 'dna'):
      d = {'sequence': e['sequence'], 'count': len(ids)}
      mods = [{'modificationType': 'CCD_' + m['modificationType'],
               'basePosition': m['basePosition']}
              for m in e.get('modifications', [])]
      if mods:
        d['modifications'] = mods
      seqs.append({'rnaSequence' if kind == 'rna' else 'dnaSequence': d})
    elif kind == 'ligand':
      if 'smiles' in e:
        seqs.append({'ligand': {'ligand': e['smiles'], 'count': len(ids)}})
      elif len(e['ccdCodes']) == 1 and e['ccdCodes'][0].upper() in (
          'MG', 'ZN', 'CA', 'NA', 'K', 'CL', 'MN', 'FE', 'CU', 'CO', 'NI'):
        seqs.append({'ion': {'ion': e['ccdCodes'][0], 'count': len(ids)}})
      else:
        seqs.append({'ligand': {'ligand': '_'.join(
            ['CCD'] + list(e['ccdCodes'])), 'count': len(ids)}})
    else:
      raise NotImplementedError(kind)
  out = {'sequences': seqs, 'name': j['name']}
  if j.get('bondedAtomPairs'):
    out['covalent_bonds'] = [
        {'entity1': entity_of[a[0]], 'position1': a[1], 'atom1': a[2],
         'entity2': entity_of[b[0]], 'position2': b[1], 'atom2': b[2]}
        for a, b in j['bondedAtomPairs']]
  return [out]


def _has_msa(out_dir):
  with open(os.path.join(out_dir, 'opendde.json')) as f:
    job, = json.load(f)
  return any('unpairedMsa' in e.get('proteinChain', {}) or
             'pairedMsa' in e.get('proteinChain', {}) for e in job['sequences'])


def _configs(model, out_dir, dtype='fp32', n_cycle=None, n_sample=None):
  from opendde.config.inference import build_inference_config
  # --use_msa follows the INPUT, as `opendde pred` would run it. With no
  # alignment anywhere, OpenDDE's single-sequence mode is --use_msa false:
  # with true, `opendde pred` would first SEARCH for one (runner/msa_search.py),
  # and `runner/inference.py` handed a bare chain builds a paired-query row plus
  # an unpaired-query row (InferenceMSAFeaturizer -> FeatureAssemblyLine, two
  # live copies of the query) -- a configuration the CLI never runs. False takes
  # make_dummy_feature's depth-1 query MSA, the one OpenDDE inference sees
  # offline. (Measured 2026-10-19, 6MRR, one pass: stock with the two-row MSA
  # vs the same with one row, trunk pair max|d|/rms 1.05.)
  # OpenDDE's own merge (base + data + inference defaults + model registry,
  # then the CLI args again on top), driven by an arg string as its CLI is.
  # Kernels are an implementation choice, not the model: torch everywhere
  # (no cuequivariance; OpenDDE has no deepspeed path).
  args = ('--model_name %s --input_json_path %s --dump_dir %s '
          '--load_checkpoint_path %s --dtype %s --triangle_attention torch '
          '--triangle_multiplicative torch --seeds 1 --use_msa %s'
          % (NAMES[model], os.path.join(out_dir, 'opendde.json'), out_dir,
             CKPT, dtype, 'true' if _has_msa(out_dir) else 'false'))
  # TF32: OpenDDE's inference defaults turn it on (inference_defaults.py,
  # enable_tf32=True -> OpenDDE.__init__ sets torch.backends.cuda.matmul /
  # cudnn.allow_tf32), so on a GPU every matmul rounds to a 10-bit mantissa
  # while ours runs at 'highest'. A precision setting, not the model: off for
  # the strict comparison. OPENDDE_TF32=1 keeps it.
  if not os.environ.get('OPENDDE_TF32'):
    args += ' --enable_tf32 false'
  # N_cycle is read by OpenDDE.__init__ (self.N_cycle), so it has to be in the
  # config the runner builds the model from -- set through the CLI key.
  if n_cycle is not None:
    args += ' --model.N_cycle %d' % n_cycle
  if n_sample is not None:
    args += ' --sample_diffusion.N_sample %d' % n_sample
  if os.path.isdir(_template_dir(out_dir)):
    args += ' --use_template true'
  c = build_inference_config(arg_str=args, fill_required_with_null=True)
  if os.path.isdir(_template_dir(out_dir)):
    # offline template caches, as `opendde pred --use_template true` would use downloaded ones
    import shutil
    tdir = _template_dir(out_dir)
    c.data.template.prot_template_mmcif_dir = os.path.join(tdir, 'mmcif')
    c.data.template.release_dates_path = os.path.join(tdir, 'release_dates.json')
    c.data.template.obsolete_pdbs_path = os.path.join(tdir, 'obsolete.json')
    c.data.template.kalign_binary_path = shutil.which('kalign') or os.path.expanduser('~/.local/bin/kalign')
    c.data.template.fetch_remote = False
  return c


def _batch(c):
  from opendde.data.inference.infer_dataloader import get_inference_dataloader
  data, atom_array, err = next(iter(get_inference_dataloader(configs=c)))[0]
  if err:
    raise RuntimeError('opendde featurisation: %s' % err)
  return data, atom_array


def featurise(model, case_json, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  with open(os.path.join(out_dir, 'opendde.json'), 'w') as f:
    json.dump(_opendde_json(j, out_dir), f)
  c = _configs(model, out_dir)
  # Seeded as forward seeds it, then the draw is SAVED: ref_pos is a random
  # draw and forward re-featurises with the same seed, so the two agree.
  from opendde.utils.seed import seed_everything
  seed_everything(seed=1, deterministic=c.deterministic)
  data, atom_array = _batch(c)
  f = data['input_feature_dict']
  names = list(atom_array.atom_name)
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names),
           atom_token=f['atom_to_token_idx'].numpy(),
           atom_pad=np.ones(len(names), bool),
           token_pad=np.ones(f['residue_index'].shape[-1], bool),
           **{'feat.ref_pos': f['ref_pos'].numpy(),
              'feat.ref_space_uid': f['ref_space_uid'].numpy()},
           # the template features OpenDDE built (for a field-by-field diff against ours)
           **{'feat.' + k: np.asarray(v) for k, v in f.items()
              if k.startswith('template') and hasattr(v, 'shape')})


def forward(model, out_dir, in_npz, out_npz):
  import torch
  from opendde.utils.seed import seed_everything
  from runner.inference import InferenceRunner, update_inference_configs
  z = np.load(in_npz)
  dtype = 'bf16' if str(z['precision']) == 'bf16' else 'fp32'
  c = _configs(model, out_dir, dtype, n_cycle=int(z['recycling_steps']) + 1)
  runner = InferenceRunner(c)
  c = runner.configs
  # As runner/inference.py:infer_predict does it: model already built, then
  # seed_everything(seed, deterministic=configs.deterministic), THEN featurise
  # and predict in the same stream (the MSA module's per-recycle row
  # permutation draws from it). Same seed as `featurise`, so this ref_pos draw
  # is the one the harness already injected into ours.
  seed_everything(seed=1, deterministic=c.deterministic)
  data, _ = _batch(c)
  data['input_feature_dict']['inference_seed'] = torch.tensor(1, dtype=torch.long)
  runner.update_model_configs(update_inference_configs(
      c, data['N_token'].item()))
  net = runner.model
  dev = runner.device
  pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32, device=dev)
  sigma = float(z['noise_level'])
  got = {}

  # THE override: OpenDDE.sample_diffusion (the wrapper the stage calls) ->
  # one real denoiser call on the injected state, with exactly the arguments
  # generator.sample_diffusion passes it, and the injected coordinates back.
  def sample_diffusion(attn_chunk_size=None, diffusion_chunk_size=None, **kw):
    s_trunk = kw['s_trunk']
    got['denoised'] = kw['denoise_net'](
        x_noisy=noisy[None].to(s_trunk.dtype),
        t_hat_noise_level=torch.full((1,), sigma, device=dev,
                                     dtype=s_trunk.dtype),
        input_feature_dict=kw['input_feature_dict'],
        s_inputs=kw['s_inputs'], s_trunk=s_trunk, z_trunk=kw['z_trunk'],
        pair_z=kw['pair_z'], pair_z_spec=kw.get('pair_z_spec'),
        p_lm=kw['p_lm'], c_l=kw['c_l'],
        atom_window_spec=kw.get('atom_window_spec'),
        foldcp_attention_bias=kw.get('foldcp_attention_bias'),
        chunk_size=attn_chunk_size, inplace_safe=False,
        enable_efficient_fusion=kw.get('enable_efficient_fusion', False),
    ).detach().clone()
    return pos[None]
  net.sample_diffusion = sample_diffusion

  orig_trunk = net.get_pairformer_output

  def get_pairformer_output(*a, **kw):
    s_inputs, s, zz = orig_trunk(*a, **kw)
    # CLONED: OpenDDE runs inplace_safe=True and may update these in place
    # afterwards; an un-cloned capture is a moving reference.
    got['s_inputs'], got['s'], got['z'] = (
        s_inputs.detach().clone(), s.detach().clone(), zz.detach().clone())
    return s_inputs, s, zz
  net.get_pairformer_output = get_pairformer_output
  net.distogram_head.register_forward_hook(
      lambda _m, _i, o: got.setdefault('distogram_logits', o.detach().clone()))

  # Stage taps: every top-level submodule's outputs, for localisation.
  taps, calls = {}, {}

  def hook(name):
    def f(_m, _i, o):
      n = calls[name] = calls.get(name, -1) + 1
      tag = name if n == 0 else '%s@%d' % (name, n)
      for k, x in enumerate(o if isinstance(o, (tuple, list)) else (o,)):
        if hasattr(x, 'float'):
          taps['tap.%s.%d' % (tag, k)] = x.detach().float().cpu().numpy().copy()
    return f
  for name, m in net.named_children():
    m.register_forward_hook(hook(name))

  pred = runner.predict(data)
  full = pred.get('full_data', [{}])
  full = full[0] if isinstance(full, list) else full
  out = {**got,
         'pae_logits': pred['pae'], 'pde_logits': pred['pde'],
         'plddt_logits': pred['plddt']}
  if 'structural_pae' in pred:
    out['structural_pae_logits'] = pred['structural_pae']
  for k in ('token_pair_pae', 'token_pair_pde', 'atom_plddt'):
    if k in full:
      out[{'token_pair_pae': 'pae', 'token_pair_pde': 'pde',
           'atom_plddt': 'plddt'}[k]] = full[k]
  out.update(taps)
  np.savez(out_npz, **{k: v.float().cpu().numpy().squeeze()
                       if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in out.items()})


def trajectory(model, out_dir, in_npz, out_npz):
  """OpenDDE's REAL sampler (generator.sample_diffusion, 200 steps, the last
  sigma 0), one sample, every random draw the harness's (vendor_native_parity
  --trajectory): its torch.randn calls (initial positions, per-step noise; they
  take OpenDDE's private generator) and centre_random_augmentation (scipy
  rotation from its numpy rng, applied R @ x -- the replacement applies the
  harness's x @ R)."""
  import torch
  from opendde.utils.seed import seed_everything
  from runner.inference import InferenceRunner, update_inference_configs
  from opendde.model import generator as G
  z = np.load(in_npz)
  c = _configs(model, out_dir, 'fp32', n_cycle=int(z['recycling_steps']) + 1, n_sample=1)
  runner = InferenceRunner(c)
  c = runner.configs
  seed_everything(seed=1, deterministic=c.deterministic)
  data, _ = _batch(c)
  data['input_feature_dict']['inference_seed'] = torch.tensor(1, dtype=torch.long)
  runner.update_model_configs(update_inference_configs(c, data['N_token'].item()))
  net = runner.model
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32)
  n = {'randn': 0, 'aug': 0}
  got = {}

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def randn(self, size=None, device=None, dtype=None, generator=None, **kw):
      k = n['randn']
      n['randn'] += 1
      v = x0 if k == 0 else eps[k - 1]
      return v.reshape(size).to(device=device, dtype=dtype or torch.float32)

  def aug(x_input_coords, N_sample=1, s_trans=1.0, centre_only=False, mask=None, eps=1e-12,
          torch_generator=None, numpy_rng=None, **kw):
    assert N_sample == 1 and not centre_only and mask is None
    k = n['aug']
    n['aug'] += 1
    x = x_input_coords - x_input_coords.mean(dim=-2, keepdim=True)
    return (x @ rot[k].to(x) + s_trans * trans[k].to(x))[..., None, :, :]

  G.torch, G.centre_random_augmentation = _Torch(), aug
  orig = net.sample_diffusion

  def sample_diffusion(*a, **kw):
    got['sigmas'] = kw['noise_schedule'].detach().float().cpu().numpy()
    out = orig(*a, **kw)
    got.setdefault('final', out.detach().float().cpu().numpy().reshape(-1, 3))
    return out
  net.sample_diffusion = sample_diffusion
  runner.predict(data)
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
