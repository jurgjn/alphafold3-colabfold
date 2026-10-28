"""The STOCK Mol-1 half of vendor_native_parity.py (mol1).

  python mol1_upstream_runner.py featurise  mol1 CASE.json OUT_DIR
  python mol1_upstream_runner.py forward    mol1 OUT_DIR IN.npz OUT.npz
  python mol1_upstream_runner.py trajectory mol1 OUT_DIR IN.npz OUT.npz

Run in ~/venv_mol1 (Molence's Mol-1 installed from ~/vendor_src/Mol-1). Mol-1
forks Protenix at bd54a05, so this is protenix_upstream_runner.py's protocol on
Mol-1's own modules: configs merged the way `mol/runner/inference.py:run` merges
them, the batch from `get_inference_dataloader`, the model and checkpoint through
`InferenceRunner`, the forward through `InferenceRunner.predict`. The input JSON
dialect is Protenix's, so the case conversion is shared.

Settings, each Mol-1's own inference default unless an env var says otherwise:
  TF32 off (enable_tf32 False); MOL1_TF32=1 turns it on.
  MC dropout off (mc_dropout_apply_rate 0.0) -- unlike protenix, nothing to replay.
  Diffusion conditioning cached in 768-row chunks
    (infer_setting.diffusion_conditioning_row_chunk_size). That path runs
    OUTSIDE the autocast-disable the unchunked one has, so under bf16 it
    computes in bf16 where Protenix computes fp32. MOL1_COND_ROW_CHUNK=0 runs
    the unchunked (fp32) path; in fp32 runs the two are the same.
  MOL1_CHUNK=N sets infer_setting.chunk_size (attention rows per chunk; memory only).
Kernels are an implementation choice, not the model: torch LayerNorm and torch
triangle kernels, as for protenix.
"""
import os as _os, sys as _sys  # dev/af3_dev_paths.py: vendor trees, venvs and weights
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from af3_dev_paths import dev_path  # noqa: E402
import json
import os
import sys

os.environ.setdefault('LAYERNORM_TYPE', 'torch')
os.environ.setdefault('MOL_ROOT_DIR', dev_path('~/mol1_data'))

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protenix_upstream_runner as PX  # noqa: E402  (the case -> input JSON conversion)

CKPT = os.path.expanduser(os.environ.get('MOL1_CKPT', dev_path('~/mol1_weights/mol-1-base.pt')))
MODEL_NAME = 'mol-1-base'


def _template_a3m(e, cid, out_dir):
  """protenix_upstream_runner._template_a3m with Mol-1's own template parser."""
  import mol.data.template.template_parser as TP
  import types
  sys.modules.setdefault('protenix', types.ModuleType('protenix'))
  for name, mod in (('protenix.data', None), ('protenix.data.template', None),
                    ('protenix.data.template.template_parser', TP)):
    sys.modules.setdefault(name, mod or types.ModuleType(name))
  return _ORIG_TEMPLATE_A3M(e, cid, out_dir)


_ORIG_TEMPLATE_A3M = PX._template_a3m
PX._template_a3m = _template_a3m


def _configs(json_path, dump_dir, dtype):
  from collections.abc import Mapping
  from mol.config.base import configs as configs_base
  from mol.config.data import data_configs
  from mol.config.inference import inference_configs
  from mol.config.model_type import model_configs
  from mol.config.parser import parse_configs

  base = {**configs_base, **{'data': data_configs}, **inference_configs}

  def deep_update(d, u):
    for k, v in u.items():
      if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
        deep_update(d[k], v)
      else:
        d[k] = v
    return d
  deep_update(base, model_configs[MODEL_NAME])
  args = ('--model_name %s --input_json_path %s --dump_dir %s --load_checkpoint_path %s '
          '--dtype %s --triangle_attention torch --triangle_multiplicative torch '
          '--seeds 1 --use_msa true' % (MODEL_NAME, json_path, dump_dir, CKPT, dtype))
  if os.environ.get('MOL1_TF32'):
    args += ' --enable_tf32 true'
  c = parse_configs(configs=base, arg_str=args, fill_required_with_null=True)
  if os.environ.get('MOL1_COND_ROW_CHUNK') == '0':
    c.infer_setting.diffusion_conditioning_row_chunk_size = None
  if os.environ.get('MOL1_CHUNK'):
    # attention chunk rows: memory only (the 1024 default does not fit long_chain on a 22 GB card)
    c.infer_setting.chunk_size = int(os.environ['MOL1_CHUNK'])
  tdir = PX._template_dir(dump_dir)
  if os.path.isdir(tdir):
    import shutil
    c.use_template = True
    c.data.template.prot_template_mmcif_dir = os.path.join(tdir, 'mmcif')
    c.data.template.release_dates_path = os.path.join(tdir, 'release_dates.json')
    c.data.template.obsolete_pdbs_path = os.path.join(tdir, 'obsolete.json')
    c.data.template.kalign_binary_path = shutil.which('kalign') or dev_path('~/.local/bin/kalign')
    c.data.template.fetch_remote = False
  return c


def _batch(out_dir, dtype='fp32'):
  from mol.data.inference.infer_dataloader import get_inference_dataloader
  c = _configs(os.path.join(out_dir, 'protenix.json'), out_dir, dtype)
  data, atom_array, err = next(iter(get_inference_dataloader(configs=c)))[0]
  if err:
    raise RuntimeError('Mol-1 featurisation: %s' % err)
  return c, data, atom_array


def featurise(model, case_json, out_dir):
  import torch
  assert model == 'mol1', model
  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  with open(os.path.join(out_dir, 'protenix.json'), 'w') as f:
    json.dump(PX._protenix_json(j, os.path.join(out_dir, 'msa')), f)
  # seeded, then SAVED: ref_pos is a random draw and forward must see this one
  from mol.utils.seed import seed_everything
  seed_everything(seed=1, deterministic=True)
  _, data, atom_array = _batch(out_dir)
  torch.save(data, os.path.join(out_dir, 'data.pt'))
  f = data['input_feature_dict']
  names = list(atom_array.atom_name)
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names),
           atom_token=f['atom_to_token_idx'].numpy(),
           atom_pad=np.ones(len(names), bool),
           token_pad=np.ones(f['residue_index'].shape[-1], bool),
           **{'feat.ref_pos': f['ref_pos'].numpy(),
              'feat.ref_space_uid': f['ref_space_uid'].numpy(),
              **{'feat.' + k: f[k].float().numpy() for k in ('msa', 'has_deletion', 'deletion_value', 'profile',
                                                            'deletion_mean') if k in f}},
           **{'feat.' + k: np.asarray(v) for k, v in f.items()
              if k.startswith('template') and hasattr(v, 'shape')})


def _runner(out_dir, dtype, recycling_steps):
  from mol.runner.inference import InferenceRunner, update_inference_configs
  from mol.utils.seed import seed_everything
  c = _configs(os.path.join(out_dir, 'protenix.json'), out_dir, dtype)
  c.model.N_cycle = int(recycling_steps) + 1
  return c, InferenceRunner, update_inference_configs, seed_everything


def forward(model, out_dir, in_npz, out_npz):
  import torch
  assert model == 'mol1', model
  z = np.load(in_npz)
  dtype = 'bf16' if str(z['precision']) == 'bf16' else 'fp32'
  c, InferenceRunner, update_inference_configs, seed_everything = _runner(out_dir, dtype, z['recycling_steps'])
  runner = InferenceRunner(c)
  # as mol/runner/inference.py infer_predict: the model built, seed_everything(seed), THEN featurise and predict
  seed_everything(seed=1, deterministic=True)
  _, data, _ = _batch(out_dir, dtype)
  runner.update_model_configs(update_inference_configs(c, data['N_token'].item()))
  net = runner.model
  dev = runner.device
  pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32, device=dev)
  sigma = float(z['noise_level'])
  got = {}

  def sample_diffusion(denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
                       N_sample, noise_schedule, inplace_safe=False, enable_efficient_fusion=False,
                       projected_pair_biases=None, **_):
    kw = {} if projected_pair_biases is None else {'projected_pair_biases': projected_pair_biases}
    got['denoised'] = denoise_net(
        x_noisy=noisy[None].to(s_trunk.dtype),
        t_hat_noise_level=torch.full((1,), sigma, device=dev, dtype=s_trunk.dtype),
        input_feature_dict=input_feature_dict, s_inputs=s_inputs,
        s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l,
        inplace_safe=False, enable_efficient_fusion=enable_efficient_fusion, **kw).detach().clone()
    return pos[None]
  net.sample_diffusion = sample_diffusion

  orig_trunk = net.get_pairformer_output

  def get_pairformer_output(*a, **kw):
    s_inputs, s, zz = orig_trunk(*a, **kw)
    # cloned: Mol-1 runs inplace_safe and updates these afterwards
    got['s_inputs'], got['s'], got['z'] = (s_inputs.detach().clone(), s.detach().clone(), zz.detach().clone())
    return s_inputs, s, zz
  net.get_pairformer_output = get_pairformer_output
  net.distogram_head.register_forward_hook(
      lambda _m, _i, o: got.setdefault('distogram_logits', o.detach().clone()))

  # stage taps: every top-level submodule's outputs, first call as tap.NAME.k, later ones tap.NAME@n.k
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

  # the MSA rows native samples on each pass, so the harness can check our replay row for row
  import mol.model.utils as MU
  orig_si = MU.sample_indices
  got_rows = []

  def sample_indices(*a, **kw):
    out = orig_si(*a, **kw)
    got_rows.append(out.detach().cpu().numpy().copy())
    return out
  MU.sample_indices = sample_indices
  pred = runner.predict(data)
  MU.sample_indices = orig_si
  for k, r in enumerate(got_rows):
    got['msa_rows_pass%d' % k] = r
  full = pred.get('full_data', [{}])
  full = full[0] if isinstance(full, list) else full
  out = {**got, 'pae_logits': pred['pae'], 'pde_logits': pred['pde'], 'plddt_logits': pred['plddt']}
  for k in ('token_pair_pae', 'token_pair_pde', 'atom_plddt'):
    if k in full:
      out[{'token_pair_pae': 'pae', 'token_pair_pde': 'pde', 'atom_plddt': 'plddt'}[k]] = full[k]
  out.update(taps)
  np.savez(out_npz, **{k: v.float().cpu().numpy().squeeze() if hasattr(v, 'cpu') else np.asarray(v).squeeze()
                       for k, v in out.items()})


def trajectory(model, out_dir, in_npz, out_npz):
  """Mol-1's REAL sampler at its inference defaults (200 steps), one sample, every random draw the
  harness's: the initial positions and per step the augmentation and the noise eps. Mol-1 rotates as
  R @ x (rot_vec_mul) where the harness's draw means x @ R, so it gets the transpose."""
  import torch
  from mol.model import generator as G
  assert model == 'mol1', model
  z = np.load(in_npz)
  c, InferenceRunner, update_inference_configs, seed_everything = _runner(out_dir, 'fp32', z['recycling_steps'])
  c.sample_diffusion.N_sample = 1
  runner = InferenceRunner(c)
  seed_everything(seed=1, deterministic=True)
  _, data, _ = _batch(out_dir, 'fp32')
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
    return torch.einsum('ij,...j->...i', rot[k], x) + trans[k] * s_trans

  G.torch, G.centre_random_augmentation = _Torch(), aug
  orig = net.sample_diffusion

  def sample_diffusion(**kw):
    got['sigmas'] = kw['noise_schedule'].detach().float().cpu().numpy()
    nudge = float(os.environ.get('TRAJ_TRUNK_NUDGE', 0) or 0)
    if nudge:
      g = torch.Generator(device=kw['s_trunk'].device).manual_seed(7)
      for k in ('s_trunk', 'z_trunk', 'pair_z'):
        if torch.is_tensor(kw.get(k)):
          kw[k] = kw[k] * (1 + nudge * torch.randn(kw[k].shape, generator=g, device=kw[k].device, dtype=kw[k].dtype))
      # the per-block pair biases were projected from the un-nudged pair_z: rebuild them from the nudged one
      if kw.get('projected_pair_biases') is not None:
        kw['projected_pair_biases'] = net._prepare_projected_pair_biases(kw['pair_z'])
    out = orig(**kw)
    got.setdefault('final', out.detach().float().cpu().numpy().reshape(-1, 3))
    return out
  net.sample_diffusion = sample_diffusion
  runner.predict(data)
  np.savez(out_npz, sigmas=got['sigmas'], steps=n['aug'], final=got['final'])


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward, 'trajectory': trajectory}[cmd](*rest)
