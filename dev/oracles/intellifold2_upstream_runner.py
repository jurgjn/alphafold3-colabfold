"""The STOCK IntelliFold-2 half of vendor_native_parity.py (intellifold2).

  python intellifold2_upstream_runner.py featurise intellifold2 CASE.json OUT_DIR
  python intellifold2_upstream_runner.py forward   intellifold2 OUT_DIR IN.npz OUT.npz

Run with ONLY ~/IntelliFold (tag v2.0.4) on PYTHONPATH, in ~/venv_intellifold
(see VENDORS for why that venv). The model is IntelliFold-2's NATIVE PyTorch
stack -- the network the `intellifold_v2.pt` checkpoint is, and what our
converter reads -- driven exactly as its PyTorch entry point
`run_intellifold.py:main` (== `runner/intellifold_inference.py`) drives it:
`check_inputs` + `process_inputs` on a boltz-style YAML, `get_inference_dataloader`,
the config from `v2_inference_config.get_model_config(args)` (`--model v2`),
`IntelliFold(config, generator)` + `load_state_dict` of the checkpoint, the
CLI's own batch preparation (MSA one-hot, empty templates), then
`model(input_features, diffusion_batch_size)`. (v2.0.4's `intellifold predict`
instead runs these weights converted onto AlphaFold 3's JAX engine -- itself a
port, so not the reference.)

The single override is the model's `sample_diffusion` -- the only stochastic
step -- which returns the injected coordinates and runs the real denoiser
(`diffusion_edm_forward`, the preconditioned network) once on an injected
(noised coordinates, sigma). See boltz2_upstream_runner.py for why.

ATOM LAYOUT. IntelliFold carries atoms DENSE, (token, slot<24), with
`pred_dense_atom_mask`; the model's forward packs them flat in that mask order
(`aggregate_fn_advanced`), and the sampler, the denoiser and the injected
coordinates live in that flat order. So "the vendor's flat atom order" here is
token-major, slot-ascending, real atoms only.
"""
import os as _os, sys as _sys  # dev/af3_dev_paths.py: vendor trees, venvs and weights
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from af3_dev_paths import dev_path  # noqa: E402
import json
import os
import sys

import numpy as np

CKPT = os.path.expanduser(os.environ.get('IF2_CKPT',
                                         dev_path('~/model_v2/intellifold_v2.pt')))
# IntelliFold's cache dir (`--cache`): only its CCD (ccd_v2.pkl) is read here;
# the checkpoint is passed explicitly, so nothing is downloaded.
CACHE = os.path.expanduser(os.environ.get('IF2_CACHE', dev_path('~/if2_cache')))
SEED = 1



def _vendor_na(msg):
  """The stock vendor cannot express this input: exit 3, graded N/A by the harness."""
  print('VENDOR_NA: ' + msg, file=sys.stderr)
  sys.exit(3)


def _template_dir(out_dir):
  return os.path.join(out_dir, 'templates_in')


def _template_a3m(e, cid, out_dir):
  """The case's AF3 templates as IntelliFold's template search output (the yaml `template:` path, an hmmsearch
  a3m `>pdbid_chain/start-end mol:protein length:N`, the hit aligned to the query), each mmCIF pre-placed in an
  offline mmcif dir with its release date (construct_template_featurizer's INTELLIFOLD_CACHE layout). IntelliFold's
  template code is protenix's: it re-aligns the query to the chain's full SEQRES with kalign and prefilters the hit
  (intellifold/data/template/template_utils.py)."""
  import re
  from intellifold.data.template.template_parser import TemplateParser
  tdir = _template_dir(out_dir)
  os.makedirs(os.path.join(tdir, 'mmcif'), exist_ok=True)
  os.makedirs(os.path.join(tdir, 'common'), exist_ok=True)
  dates_path = os.path.join(tdir, 'common', 'release_date_cache.json')
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
  path = os.path.join(tdir, '%s_hmmsearch.a3m' % cid)
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
  return path

def _boltz_csv(paired, unpaired):
  """A pairedMsa + unpairedMsa as ONE boltz MSA csv, exactly as `boltz
  predict`'s compute_msa writes the ColabFold server's pair: the pair a3m's
  rows keyed by row index (all-gap rows dropped), then the unpaired rows past
  their query keyed -1. boltz pairs across chains on that key."""
  pr = [l.strip() for l in paired.strip().splitlines()[1::2]]
  keys = [i for i, s in enumerate(pr) if s != '-' * len(s)]
  pr = [s for s in pr if s != '-' * len(s)]
  ur = [l.strip() for l in unpaired.strip().splitlines()[1::2]] if unpaired.strip() else []
  if pr:
    ur = ur[1:]
  return '\n'.join(['key,sequence'] + ['%s,%s' % (k, q) for k, q in
                                        zip(keys + [-1] * len(ur), pr + ur)])


def _yaml_from_af3(j, msa_dir):
  """AF3 fold-input JSON -> IntelliFold YAML (docs/input_yaml_format.md, the
  boltz schema). A chain whose alignment is only its own query runs the
  single-sequence path the CLI documents (`msa: empty`). An AF3 ligand -- Mg
  included -- is a CCD ligand (`ccd:`); the YAML has no separate ion type."""
  import yaml
  seqs = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    if kind == 'protein':
      d = {'id': ids, 'sequence': e['sequence']}
      if e.get('cyclic'):
        d['cyclic'] = True
      msa = e.get('unpairedMsa')
      if len((e.get('pairedMsa') or '').splitlines()) > 2:
        p = os.path.join(msa_dir, '%s.csv' % ids[0])
        with open(p, 'w') as f:
          f.write(_boltz_csv(e['pairedMsa'], msa or ''))
        d['msa'] = p
      elif ((not msa) or msa.strip() == '>q\n%s' % e['sequence']) and not e.get('templates'):
        d['msa'] = 'empty'
      elif (not msa) or msa.strip() == '>q\n%s' % e['sequence']:
        # a template is only reachable beside a parsed MSA (data_tools.process_inputs copies chain i's template
        # to processed/templates as templates[msa_idx]; with `msa: empty` its templatesPath points at nothing),
        # so a templated chain without alignments gets its query as a one-row a3m
        p = os.path.join(msa_dir, '%s.a3m' % ids[0])
        with open(p, 'w') as f:
          f.write('>query\n%s\n' % e['sequence'])
        d['msa'] = p
      else:
        p = os.path.join(msa_dir, '%s.a3m' % ids[0])
        with open(p, 'w') as f:
          f.write(msa)
        d['msa'] = p
      mods = [{'position': m['ptmPosition'], 'ccd': m['ptmType']}
              for m in e.get('modifications', [])]
      if mods:
        d['modifications'] = mods
      if e.get('templates'):
        d['template'] = _template_a3m(e, ids[0], msa_dir)
    elif kind in ('rna', 'dna'):
      d = {'id': ids, 'sequence': e['sequence']}
      mods = [{'position': m['basePosition'], 'ccd': m['modificationType']}
              for m in e.get('modifications', [])]
      if mods:
        d['modifications'] = mods
    elif kind == 'ligand':
      d = {'id': ids}
      if 'smiles' in e:
        d['smiles'] = e['smiles']
      else:
        if len(e['ccdCodes']) > 1:   # the boltz schema's ccd list: a multi-residue ligand
          d['ccd'] = e['ccdCodes']
          seqs.append({kind: d})
          continue
        d['ccd'] = e['ccdCodes'][0]
    else:
      raise NotImplementedError(kind)
    seqs.append({kind: d})
  y = {'version': 1, 'sequences': seqs}
  if j.get('bondedAtomPairs'):
    y['constraints'] = [{'bond': {'atom1': list(a), 'atom2': list(b)}}
                        for a, b in j['bondedAtomPairs']]
  return yaml.safe_dump(y, sort_keys=False)


def _args(recycles=10, precision='no'):
  """`run_intellifold.py`'s argparse namespace at its defaults, but `--model
  v2` (the checkpoint ours is converted from; the CLI default is v2-flash) and
  these deliberate departures:
    precision   'no' (fp32) unless the harness asks for bf16 -- the CLI's bf16
                default is a precision choice, not the model;
    num_workers 0 -- the draw then comes from the seeded main process; a worker
                count only changes WHICH random ref_pos draw is taken, and the
                harness injects whichever one this is into ours;
    num_diffusion_samples 1 -- one sample is scored."""
  from types import SimpleNamespace
  return SimpleNamespace(
      out_dir='./', cache=CACHE, num_workers=0, precision=precision,
      seed=str(SEED), recycling_iters=recycles, num_diffusion_samples=1,
      sampling_steps=200, output_format='mmcif', override=True,
      use_msa_server=False, msa_server_url='https://api.colabfold.com',
      msa_pairing_strategy='greedy', no_pairing=False, use_template=False,
      only_run_data_process=False, return_similar_seq=False, model='v2')


def _setup():
  import torch
  # TF32 is a precision setting, not the model: IntelliFold sets nothing, so a
  # GPU run would use torch's defaults; pinned off so a GPU run is the same
  # fp32 comparison as the CPU one.
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  # Kernels are an implementation choice, not the model: IntelliFold's torch
  # LayerNorm (LAYERNORM_TYPE unset, not 'fast_layernorm') and torch attention
  # (USE_DEEPSPEED_EVO_ATTENTION unset; deepspeed is not installed).
  os.environ.pop('LAYERNORM_TYPE', None)
  os.environ.pop('USE_DEEPSPEED_EVO_ATTENTION', None)
  os.environ['INTELLIFOLD_CACHE'] = CACHE


def _pipeline(out_dir, args, load_weights):
  """run_intellifold.py:main, in its order, up to the prepared batch:
  set_seed -> Accelerator -> process_inputs -> dataloader -> config ->
  generator -> IntelliFold(config) (whose init consumes the torch RNG) ->
  checkpoint -> prepare -> first batch (the featuriser's random ref_pos
  roto-translation draws here) -> the CLI's batch preparation."""
  import torch
  import torch.nn.functional as F
  from pathlib import Path
  from accelerate import Accelerator
  from accelerate.utils import set_seed
  from intellifold.openfold.v2_inference_config import get_model_config
  from intellifold.openfold.model.model import IntelliFold
  from intellifold.data.module.inference import (
      get_inference_dataloader, construct_empty_template_features)
  from intellifold.data.types import Manifest
  from intellifold.data.inference.params import BoltzProcessedInput
  from intellifold.data.inference.data_tools import check_inputs, process_inputs

  out = Path(out_dir)
  use_template = os.path.isdir(_template_dir(out_dir))
  if use_template:
    # `--use_template` with offline caches: the featuriser built as construct_template_featurizer builds it,
    # pointed at the case's mmcif dir and release dates instead of $INTELLIFOLD_CACHE (kalign from PATH)
    import shutil
    import intellifold.data.module.inference as ifi
    from intellifold.data.template.template_utils import TemplateHitFeaturizer
    tdir = _template_dir(out_dir)

    def construct_template_featurizer():
      return TemplateHitFeaturizer(
          mmcif_dir=os.path.join(tdir, 'mmcif'), template_cache_dir='', max_hits=4,
          kalign_binary_path=shutil.which('kalign') or dev_path('~/.local/bin/kalign'),
          max_template_date='2021-09-30',
          release_dates_path=os.path.join(tdir, 'common', 'release_date_cache.json'),
          obsolete_pdbs_path=os.path.join(tdir, 'common', 'obsolete_to_successor.json'),
          _shuffle_top_k_prefiltered=None, _max_template_candidates_num=20)
    ifi.construct_template_featurizer = construct_template_featurizer
    args.use_template = True
  set_seed(SEED)
  accelerator = Accelerator(mixed_precision=args.precision,
                            step_scheduler_with_optimizer=False)
  data = check_inputs(out / 'case.yaml')
  process_inputs(args, data=data, out_dir=out,
                 ccd_path=Path(CACHE) / 'ccd_v2.pkl', use_msa_server=False,
                 msa_server_url=args.msa_server_url,
                 msa_pairing_strategy=args.msa_pairing_strategy,
                 max_msa_seqs=16384, use_pairing=not args.no_pairing,
                 use_template=use_template)
  proc = out / 'processed'
  processed = BoltzProcessedInput(
      manifest=Manifest.load(proc / 'manifest.json'),
      targets_dir=proc / 'structures', msa_dir=proc / 'msa',
      template_dir=(proc / 'templates') if use_template else None,
      constraints_dir=(proc / 'constraints')
      if (proc / 'constraints').exists() else None)
  loader = get_inference_dataloader(
      args=args, manifest=processed.manifest,
      target_dir=processed.targets_dir, msa_dir=processed.msa_dir,
      template_dir=processed.template_dir, constraints_dir=processed.constraints_dir)
  config = get_model_config(args)
  if os.environ.get('MSA_ALL_ROWS'):
    # IntelliFold's MSAEmbedder subsamples min(msa_depth=1024, rows) rows at random each pass
    # (SampleRandomWithoutReplacement, torch.randperm): a draw, which the harness lifts on every side --
    # all rows, as ours takes them (the MSA stack is row-permutation invariant, so the randperm left in
    # place only reorders them)
    config.backbone.msa.msa_embedder.msa_depth = 1 << 20
  generator = torch.Generator(device=accelerator.device)
  generator.manual_seed(SEED)
  model = IntelliFold(config, generator=generator)
  if load_weights:
    model.load_state_dict(torch.load(CKPT, map_location=accelerator.device))
    model = accelerator.prepare(model)
    model.eval()
  loader = accelerator.prepare(loader)
  f = next(iter(loader))
  record = f.pop('record')[0]
  f.pop('structure')
  f['msa'] = F.one_hot(f['msa'].long(), num_classes=32).float()
  if use_template:
    # run_intellifold.py: the template aatype one-hot, else the empty template features
    f['template_aatype'] = F.one_hot(f['template_aatype'].long(), num_classes=31).float()
  else:
    f.update(construct_empty_template_features(f, device=accelerator.device))
  return model, f, accelerator


def featurise(model_name, case_json, out_dir):
  import torch
  _setup()
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as fh:
    j = json.load(fh)
  with open(os.path.join(out_dir, 'case.yaml'), 'w') as fh:
    fh.write(_yaml_from_af3(j, out_dir))
  # The model is BUILT (not loaded: loading draws nothing) so the featuriser's
  # ref_pos draw comes at the RNG state the CLI's does; forward repeats this
  # and checks it gets the same draw.
  _, f, _ = _pipeline(out_dir, _args(), load_weights=False)
  m = f['pred_dense_atom_mask'][0].bool().cpu().numpy()        # (tok, 24)
  chars = f['ref_atom_name_chars'][0].cpu().numpy()            # (tok,24,4,64)
  chars = chars.argmax(-1)[m]                                  # (atoms, 4)
  names = [''.join(chr(int(c) + 32) for c in row).strip() for row in chars]
  tok = np.broadcast_to(np.arange(m.shape[0])[:, None], m.shape)[m]
  flat = lambda k: f[k][0].cpu().numpy()[m].copy()
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names), atom_token=tok,
           atom_pad=np.ones(len(names), bool),
           token_pad=f['seq_mask'][0].cpu().numpy().astype(bool),
           **{'feat.ref_pos': flat('ref_pos'),
              'feat.ref_space_uid': flat('ref_space_uid'),
              'feat.ref_element': flat('ref_element'),
              'feat.ref_charge': flat('ref_charge')},
           # the template features IntelliFold built (for a field-by-field diff against ours)
           **{'feat.' + k: f[k][0].float().cpu().numpy() for k in f
              if (k.startswith('template') or k.startswith('msa') or k in ('profile', 'deletion_mean',
                  'has_deletion', 'deletion_value')) and hasattr(f[k], 'shape')})


def forward(model_name, out_dir, in_npz, out_npz):
  import torch
  from accelerate.utils import set_seed
  _setup()
  z = np.load(in_npz)
  bf16 = str(z['precision']) == 'bf16'
  net, f, acc = _pipeline(
      out_dir, _args(int(z['recycling_steps']), 'bf16' if bf16 else 'no'),
      load_weights=True)
  lay = np.load(os.path.join(out_dir, 'layout.npz'))
  m = f['pred_dense_atom_mask'][0].bool().cpu().numpy()
  if not np.array_equal(f['ref_pos'][0].cpu().numpy()[m], lay['feat.ref_pos']):
    raise RuntimeError('ref_pos draw differs from the one featurise saved '
                       '(and the harness injected into ours)')
  dev = acc.device
  # As run_intellifold.py does per seed: set_seed, generator re-seeded, then
  # the forward under no_grad.
  set_seed(SEED)
  base = net.module if hasattr(net, 'module') else net
  base.generator.manual_seed(SEED)

  pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32, device=dev)
  sigma = float(z['noise_level'])
  got = {}

  # THE override: one real denoiser call (diffusion_edm_forward: the EDM
  # preconditioning around diffusion_module, exactly the call each sampler
  # step makes) on the injected state; the injected coordinates back, in the
  # flat layout sample_diffusion returns (reverse_fn densifies them after).
  def sample_diffusion(input_features, s_inputs, s_trunk, z_trunk,
                       diffusion_batch_size):
    t = torch.full((1, 1, 1), sigma, device=dev, dtype=torch.float32)
    with torch.no_grad():
      got['denoised'] = base.diffusion_edm_forward(
          x_noisy=noisy[None], t=t, input_features=input_features,
          s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk
      ).detach().clone()
    return pos[None].repeat(diffusion_batch_size, 1, 1)
  base.sample_diffusion = sample_diffusion

  # Trunk outputs, CLONED: the forward may update tensors in place later, and
  # on CPU `.float().cpu().numpy()` is a view of the same storage.
  orig_trunk = base.backbone_trunk.forward

  def trunk(feats):
    o = orig_trunk(feats)
    for k in ('s_inputs', 's', 'z', 'distogram_logits'):
      got[k] = o[k].detach().clone()
    return o
  base.backbone_trunk.forward = trunk

  # Stage taps: every trunk / diffusion / confidence submodule's outputs, for
  # localisation; repeated calls (recycling) as NAME@n.
  taps, calls = {}, {}

  def hook(name):
    def fn(_m, _i, o):
      n = calls[name] = calls.get(name, -1) + 1
      tag = name if n == 0 else '%s@%d' % (name, n)
      if isinstance(o, dict):
        items = list(o.items())
      else:
        items = list(enumerate(o if isinstance(o, (tuple, list)) else (o,)))
      for k, x in items:
        if hasattr(x, 'float'):
          taps['tap.%s.%s' % (tag, k)] = x.detach().float().cpu().numpy().copy()
    return fn
  # VENDOR_NO_TAPS: localisation only, and on long_chain the per-recycle pair taps are gigabytes of npz
  for parent in (() if os.environ.get('VENDOR_NO_TAPS') else ('backbone_trunk', 'diffusion_module', 'confidence_head')):
    for name, mod in getattr(base, parent).named_children():
      mod.register_forward_hook(hook('%s.%s' % (parent, name)))

  with torch.no_grad():
    out = net(f, diffusion_batch_size=1)
  mask = f['pred_dense_atom_mask'][0].bool()
  res = {k: v for k, v in got.items()}
  res['pae'] = out['pae']
  res['pde'] = out['pde']
  # pLDDT per ATOM, in the flat order (the dense head read at real slots).
  res['plddt'] = out['plddt'][0][mask]
  for k in ('pae_logits', 'pde_logits', 'plddt_logits', 'ptm', 'iptm'):
    res[k] = out[k]
  res.update(taps)
  np.savez(out_npz, **{k: v.detach().float().cpu().numpy().squeeze()
                       if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in res.items()})


def trajectory(model_name, out_dir, in_npz, out_npz):
  """IntelliFold's REAL sample_diffusion at its CLI default (200 steps), one
  sample, every random draw the harness's (vendor_native_parity --trajectory):
  its two `torch.empty_like(...).normal_(generator=)` draws (initial positions,
  per-step noise) and its centre_random_augmentation's rotation and
  translation. IntelliFold rotates as x @ R^T; the replacement applies the
  harness's x @ R, keeping IntelliFold's masked centring and mask multiplies."""
  import torch
  from accelerate.utils import set_seed
  import intellifold.openfold.model.model as M
  _setup()
  z = np.load(in_npz)
  net, f, acc = _pipeline(out_dir, _args(int(z['recycling_steps']), 'no'),
                          load_weights=True)
  dev = acc.device
  set_seed(SEED)
  base = net.module if hasattr(net, 'module') else net
  base.generator.manual_seed(SEED)
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32, device=dev)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32, device=dev)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32, device=dev)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32, device=dev)
  n = {'normal': 0, 'aug': 0}
  got = {}

  class _Draw:
    def __init__(self, like):
      self.like = like

    def normal_(self, generator=None):
      k = n['normal']
      n['normal'] += 1
      v = x0 if k == 0 else eps[k - 1]
      if v.numel() != self.like.numel():
        raise RuntimeError('draw %d: vendor shape %s, harness %s'
                           % (k, tuple(self.like.shape), tuple(v.shape)))
      return v.reshape(self.like.shape).to(self.like)

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def empty_like(self, x, **kw):
      return _Draw(x)

  def aug(x, mask, generator=None):
    k = n['aug']
    n['aug'] += 1
    m = mask.unsqueeze(-1)
    mean = (x * m).sum(-2, keepdim=True) / mask.float().sum(-1, keepdim=True).unsqueeze(-1).clamp(min=1.)
    x = (x - mean) * m
    return (x @ rot[k] + trans[k]) * m

  M.torch = _Torch()
  base.centre_random_augmentation.forward = aug   # a submodule: replace its forward
  got['sigmas'] = base.noise_schedule(device=dev).float().cpu().numpy()
  orig = base.sample_diffusion

  def sample_diffusion(*a, **kw):
    out = orig(*a, **kw)
    got.setdefault('final', out.detach().float().cpu().numpy().reshape(-1, 3))
    return out
  base.sample_diffusion = sample_diffusion
  with torch.no_grad():
    net(f, diffusion_batch_size=1)
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
