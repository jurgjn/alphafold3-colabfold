"""The STOCK Protenix half of vendor_native_parity.py (protenix1, protenix2).

  python protenix_upstream_runner.py featurise MODEL CASE.json OUT_DIR
  python protenix_upstream_runner.py forward   MODEL OUT_DIR IN.npz OUT.npz

Run with ONLY ~/protenix on PYTHONPATH. Everything is protenix's own: configs
merged the way `runner/inference.py:run` merges them, the batch from
`get_inference_dataloader`, the model and checkpoint through `InferenceRunner`,
the forward through `InferenceRunner.predict`. The single override is the
model's `sample_diffusion` -- the only stochastic step -- which returns the
injected coordinates and runs the real denoiser once on an injected
(noised coordinates, sigma). See boltz2_upstream_runner.py for why.
"""
import json
import os
import sys

# protenix's fused LayerNorm is a JIT-compiled CUDA kernel, chosen at import by
# this variable; like the triangle kernels below, an implementation choice, so
# the stock torch LayerNorm is the reference.
os.environ.setdefault('LAYERNORM_TYPE', 'torch')

import numpy as np

CKPT_DIR = os.path.expanduser(os.environ.get('PROTENIX_CKPT_DIR', '~/protenix_weights'))
NAMES = {'protenix2': 'protenix-v2', 'protenix1': 'protenix_base_default_v1.0.0'}



def _vendor_na(msg):
  """The stock vendor cannot express this input: exit 3, graded N/A by the harness."""
  print('VENDOR_NA: ' + msg, file=sys.stderr)
  sys.exit(3)


def _template_dir(out_dir):
  return os.path.join(out_dir, 'templates')


def _template_a3m(e, cid, out_dir):
  """The case's AF3 templates as protenix's template search output (templatesPath, an hmmsearch a3m:
  `>pdbid_chain/start-end mol:protein length:N`, the hit aligned to the query), with each template's mmCIF
  pre-placed in an offline mmcif dir (<pdbid>.cif, fetch_remote off) and its release date in a release-dates
  file, as protenix's downloaded caches would hold them. The hit line follows the case's query->template map;
  protenix itself then re-aligns the query to the template chain's full SEQRES with kalign
  (TemplateHitProcessor._align_query_to_hit_index_mapping) and prefilters the hit (TemplateHitFilter)."""
  import re
  from protenix.data.template.template_parser import TemplateParser
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

def _protenix_json(j, msa_root):
  """AF3 fold-input JSON -> protenix's input JSON (a one-element list)."""
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
      msa = e.get('unpairedMsa') or ''
      paired = e.get('pairedMsa') or ''
      has_pair = len(paired.splitlines()) > 2
      if (msa and msa.strip() != '>q\n%s' % e['sequence']) or has_pair:
        md = os.path.join(msa_root, ids[0])
        os.makedirs(md, exist_ok=True)
        with open(os.path.join(md, 'non_pairing.a3m'), 'w') as f:
          f.write(msa or '>query\n%s\n' % e['sequence'])
        # the pairedMsa as protenix's pairing.a3m: it pairs by the UniProt
        # species in each header (msa_utils.get_species_ids), as AF3 does
        with open(os.path.join(md, 'pairing.a3m'), 'w') as f:
          f.write(paired if has_pair else '>query\n%s\n' % e['sequence'])
        d['msa'] = {'precomputed_msa_dir': md, 'pairing_db': 'uniref100'}
      if e.get('templates'):
        d['templatesPath'] = _template_a3m(e, ids[0], os.path.dirname(msa_root))
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
      # An AF3 input has no `ion` type: Mg is a CCD ligand, which is protenix's
      # `ligand: CCD_MG`. Its separate `ion` entity is NOT the same input -- its
      # placeholder MSA reads class 0 in every row and the profile, where a CCD
      # ligand reads UNK (measured 2026-10-19), so translating MG to `ion` made
      # the comparison test a convention the AF3 input never asked for.
      if 'smiles' in e:
        seqs.append({'ligand': {'ligand': e['smiles'], 'count': len(ids)}})
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


def _configs(model, json_path, dump_dir, dtype):
  from collections.abc import Mapping
  from configs.configs_base import configs as configs_base
  from configs.configs_data import data_configs
  from configs.configs_inference import inference_configs
  from configs.configs_model_type import model_configs
  from protenix.config import parse_configs

  name = NAMES[model]
  base = {**configs_base, **{'data': data_configs}, **inference_configs}

  def deep_update(d, u):
    for k, v in u.items():
      if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
        deep_update(d[k], v)
      else:
        d[k] = v
    return d
  deep_update(base, model_configs[name])
  # Kernels are an implementation choice, not the model: torch everywhere.
  args = ('--model_name %s --input_json_path %s --dump_dir %s '
          '--load_checkpoint_dir %s --dtype %s --triangle_attention torch '
          '--triangle_multiplicative torch --seeds 1 --use_msa true'
          % (name, json_path, dump_dir, CKPT_DIR, dtype))
  # STOCK INFERENCE IS STOCHASTIC IN THE TRUNK: with probability
  # mc_dropout_apply_rate (0.4) per seed the recycled z update runs through
  # dropout, and each pass takes a random MSA subset. Our port REPLAYS those
  # draws from the seed (model_features._native_rng, 2136ec2), so by default
  # they stay on and the comparison tests the replay. PROTENIX_NO_RNG=1 turns
  # them off here; run ours with AF3_NO_PX_NATIVE_RNG=1 to match.
  # TF32: protenix's INFERENCE config turns it on (configs_inference.py,
  # enable_tf32=True -> torch.backends.cuda.matmul.allow_tf32), so on a GPU
  # every matmul rounds to a 10-bit mantissa while ours runs at 'highest'. A
  # precision setting, not the model: off for the strict comparison
  # (s_inputs read 3.9e-03 with it on, 1e-06 on CPU). PROTENIX_TF32=1 keeps it.
  if not os.environ.get('PROTENIX_TF32'):
    args += ' --enable_tf32 false'
  if os.environ.get('PROTENIX_NO_RNG'):
    args += ' --mc_dropout_apply_rate 0'
  c = parse_configs(configs=base, arg_str=args, fill_required_with_null=True)
  tdir = _template_dir(dump_dir)
  if os.path.isdir(tdir):
    # the CLI's `--use_template true` with offline caches (runner/batch_inference.py): kalign from PATH
    import shutil
    c.use_template = True
    c.data.template.prot_template_mmcif_dir = os.path.join(tdir, 'mmcif')
    c.data.template.release_dates_path = os.path.join(tdir, 'release_dates.json')
    c.data.template.obsolete_pdbs_path = os.path.join(tdir, 'obsolete.json')
    c.data.template.kalign_binary_path = shutil.which('kalign') or os.path.expanduser('~/.local/bin/kalign')
    c.data.template.fetch_remote = False
  return c


def _batch(model, out_dir, dtype='fp32'):
  from protenix.data.inference.infer_dataloader import get_inference_dataloader
  c = _configs(model, os.path.join(out_dir, 'protenix.json'), out_dir, dtype)
  data, atom_array, err = next(iter(get_inference_dataloader(configs=c)))[0]
  if err:
    raise RuntimeError('protenix featurisation: %s' % err)
  return c, data, atom_array


def featurise(model, case_json, out_dir):
  import torch
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  with open(os.path.join(out_dir, 'protenix.json'), 'w') as f:
    json.dump(_protenix_json(j, os.path.join(out_dir, 'msa')), f)
  # Seeded, then SAVED: ref_pos is a random draw and forward must see this one.
  from protenix.utils.seed import seed_everything
  seed_everything(seed=1, deterministic=True)
  _, data, atom_array = _batch(model, out_dir)
  torch.save(data, os.path.join(out_dir, 'data.pt'))
  f = data['input_feature_dict']
  names = list(atom_array.atom_name)
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names),
           atom_token=f['atom_to_token_idx'].numpy(),
           atom_pad=np.ones(len(names), bool),
           token_pad=np.ones(f['residue_index'].shape[-1], bool),
           **{'feat.ref_pos': f['ref_pos'].numpy(),
              'feat.ref_space_uid': f['ref_space_uid'].numpy()},
           # the template features protenix built (for a field-by-field diff against ours)
           **{'feat.' + k: np.asarray(v) for k, v in f.items()
              if k.startswith('template') and hasattr(v, 'shape')})


def forward(model, out_dir, in_npz, out_npz):
  import torch
  from runner.inference import InferenceRunner, update_inference_configs
  z = np.load(in_npz)
  dtype = 'bf16' if str(z['precision']) == 'bf16' else 'fp32'
  c = _configs(model, os.path.join(out_dir, 'protenix.json'), out_dir, dtype)
  c.model.N_cycle = int(z['recycling_steps']) + 1
  runner = InferenceRunner(c)
  # As runner/inference.py:infer_predict does it: the model already built,
  # seed_everything(seed), THEN featurise and predict in the same stream --
  # which is what the replay on our side reproduces. Same seed as `featurise`,
  # so this draw of ref_pos is the one the harness already injected into ours.
  from protenix.utils.seed import seed_everything
  seed_everything(seed=1, deterministic=True)
  _, data, _ = _batch(model, out_dir, dtype)
  runner.update_model_configs(update_inference_configs(
      c, data['N_token'].item()))
  net = runner.model
  dev = runner.device
  pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32, device=dev)
  sigma = float(z['noise_level'])
  got = {}

  def sample_diffusion(denoise_net, input_feature_dict, s_inputs, s_trunk,
                       z_trunk, pair_z, p_lm, c_l, N_sample, noise_schedule,
                       inplace_safe=False, enable_efficient_fusion=False):
    got['denoised'] = denoise_net(
        x_noisy=noisy[None].to(s_trunk.dtype),
        t_hat_noise_level=torch.full((1,), sigma, device=dev,
                                     dtype=s_trunk.dtype),
        input_feature_dict=input_feature_dict, s_inputs=s_inputs,
        s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l,
        inplace_safe=False, enable_efficient_fusion=enable_efficient_fusion
    ).detach().clone()
    return pos[None]
  net.sample_diffusion = sample_diffusion

  orig_trunk = net.get_pairformer_output

  def get_pairformer_output(*a, **kw):
    s_inputs, s, zz = orig_trunk(*a, **kw)
    # CLONED. protenix runs inplace_safe=True and updates these tensors in
    # place afterwards; on CPU `.float().cpu().numpy()` is a VIEW of the same
    # storage, so an un-cloned capture is whatever the tensor holds at the END
    # of the forward -- a "reference" that moved under the comparison.
    got['s_inputs'], got['s'], got['z'] = (
        s_inputs.detach().clone(), s.detach().clone(), zz.detach().clone())
    return s_inputs, s, zz
  net.get_pairformer_output = get_pairformer_output
  net.distogram_head.register_forward_hook(
      lambda _m, _i, o: got.setdefault('distogram_logits', o.detach().clone()))

  # Stage taps: every top-level submodule's FIRST output, for localisation.
  taps = {}

  calls = {}

  def hook(name):
    def f(_m, _i, o):
      n = calls[name] = calls.get(name, -1) + 1
      # First call as tap.NAME.k (as before); later calls as tap.NAME@n.k.
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
  for k in ('token_pair_pae', 'token_pair_pde', 'atom_plddt'):
    if k in full:
      out[{'token_pair_pae': 'pae', 'token_pair_pde': 'pde',
           'atom_plddt': 'plddt'}[k]] = full[k]
  out.update(taps)
  np.savez(out_npz, **{k: v.float().cpu().numpy().squeeze()
                       if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in out.items()})


def trajectory(model, out_dir, in_npz, out_npz):
  """protenix's REAL sampler at its inference defaults (200 steps), one sample,
  every random draw the harness's (vendor_native_parity --trajectory): the
  initial positions and per step the augmentation and the noise eps.
  protenix rotates as R @ x (rot_vec_mul) where the harness's draw means
  x @ R, so it gets the transpose."""
  import torch
  from runner.inference import InferenceRunner, update_inference_configs
  from protenix.model import generator as G
  z = np.load(in_npz)
  c = _configs(model, os.path.join(out_dir, 'protenix.json'), out_dir, 'fp32')
  c.model.N_cycle = int(z['recycling_steps']) + 1
  c.sample_diffusion.N_sample = 1
  runner = InferenceRunner(c)
  from protenix.utils.seed import seed_everything
  seed_everything(seed=1, deterministic=True)
  _, data, _ = _batch(model, out_dir, 'fp32')
  runner.update_model_configs(update_inference_configs(c, data['N_token'].item()))
  net = runner.model
  dev = runner.device
  x0 = torch.tensor(z['traj_x0'], dtype=torch.float32, device=dev)
  eps = torch.tensor(z['traj_eps'], dtype=torch.float32, device=dev)
  rot = torch.tensor(z['traj_rot'], dtype=torch.float32, device=dev).transpose(-1, -2)
  trans = torch.tensor(z['traj_trans'], dtype=torch.float32, device=dev)
  n = {'randn': 0, 'aug': 0}
  got = {}

  class _Torch:
    def __getattr__(self, k):
      return getattr(torch, k)

    def randn(self, size, device=None, dtype=None, **kw):
      k = n['randn']
      n['randn'] += 1
      v = x0 if k == 0 else eps[k - 1]
      return v.reshape(size).to(dtype or torch.float32)

  def aug(x_input_coords, N_sample=1, s_trans=1.0, centre_only=False, mask=None, eps=1e-12):
    assert N_sample == 1 and not centre_only and mask is None
    k = n['aug']
    n['aug'] += 1
    x = x_input_coords - x_input_coords.mean(dim=-2, keepdim=True)
    x = x[..., None, :, :]
    # rot_vec_mul(R, x): R @ x per point
    return torch.einsum('ij,...j->...i', rot[k], x) + trans[k] * s_trans

  G.torch, G.centre_random_augmentation = _Torch(), aug
  orig = net.sample_diffusion

  def sample_diffusion(**kw):
    got['sigmas'] = kw['noise_schedule'].detach().float().cpu().numpy()
    out = orig(**kw)
    got.setdefault('final', out.detach().float().cpu().numpy().reshape(-1, 3))
    return out
  net.sample_diffusion = sample_diffusion
  runner.predict(data)
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
