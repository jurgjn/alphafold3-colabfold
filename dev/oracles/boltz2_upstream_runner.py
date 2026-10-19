"""The STOCK Boltz-2 half of boltz2_native_parity.py. Runs in ~/venv_boltz.

  ~/venv_boltz/bin/python -I boltz2_upstream_runner.py featurise boltz2 CASE.json OUT_DIR
  ~/venv_boltz/bin/python -I boltz2_upstream_runner.py forward   boltz2 OUT_DIR IN.npz OUT.npz

Nothing here is hand-assembled. The batch comes from boltz's own pipeline
(`process_inputs` -> `Boltz2InferenceDataModule`), the model from
`Boltz2.load_from_checkpoint` with the arguments `boltz predict` passes, and the
forward is the model's own `forward`. The single override is
`structure_module.sample`, the only stochastic step: it returns the injected
coordinates and runs ONE `preconditioned_network_forward` on an injected
(noised coordinates, sigma), so a denoise step is compared on equal inputs.

Why not the hand-built adapters in denoise_parity/atom_parity/...: each of those
re-derives constructor arguments from tensor shapes, and a flag the checkpoint
sets and the class defaults differently (fix_sym_check, cyclic_pos_enc) is
silently wrong -- a fake 5.8 A boltz2 "break" on 2026-10-18. And not
~/BoltzDesign1: that is a fork with twelve modified files.
"""
import json
import os
import sys

import numpy as np

CACHE = os.path.expanduser('~/.boltz')


def _yaml_from_af3(j, msa_dir):
  """AF3 fold-input JSON -> boltz YAML text (the schema boltz documents)."""
  import yaml
  seqs = []
  for entry in j['sequences']:
    (kind, e), = entry.items()
    ids = e['id'] if isinstance(e['id'], list) else [e['id']]
    if kind == 'protein':
      d = {'id': ids, 'sequence': e['sequence']}
      msa = e.get('unpairedMsa')
      seq_only = (not msa) or msa.strip() == '>q\n%s' % e['sequence']
      if seq_only:
        d['msa'] = 'empty'
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
        raise NotImplementedError('templates: boltz reads a cif path per chain')
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
        assert len(e['ccdCodes']) == 1, 'multi-component ligand'
        d['ccd'] = e['ccdCodes'][0]
    else:
      raise NotImplementedError(kind)
    seqs.append({kind: d})
  y = {'version': 1, 'sequences': seqs}
  if j.get('bondedAtomPairs'):
    y['constraints'] = [{'bond': {'atom1': list(a), 'atom2': list(b)}}
                        for a, b in j['bondedAtomPairs']]
  return yaml.safe_dump(y, sort_keys=False)


def featurise(model, case_json, out_dir):
  import torch
  from boltz.main import process_inputs, BoltzProcessedInput
  from boltz.data.types import Manifest
  from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
  from pathlib import Path

  os.makedirs(out_dir, exist_ok=True)
  with open(case_json) as f:
    j = json.load(f)
  yp = os.path.join(out_dir, '%s.yaml' % j['name'])
  with open(yp, 'w') as f:
    f.write(_yaml_from_af3(j, out_dir))
  out = Path(out_dir)
  process_inputs(data=[Path(yp)], out_dir=out, ccd_path=Path(CACHE) / 'ccd.pkl',
                 mol_dir=Path(CACHE) / 'mols', msa_server_url='',
                 msa_pairing_strategy='greedy', boltz2=True)
  proc = out / 'processed'
  manifest = Manifest.load(proc / 'manifest.json')
  dm = Boltz2InferenceDataModule(
      manifest=manifest, target_dir=proc / 'structures', msa_dir=proc / 'msa',
      mol_dir=Path(CACHE) / 'mols', num_workers=0,
      constraints_dir=(proc / 'constraints') if (proc / 'constraints').exists() else None,
      template_dir=None,
      extra_mols_dir=(proc / 'mols') if (proc / 'mols').exists() else None)
  feats = next(iter(dm.predict_dataloader()))
  torch.save(feats, os.path.join(out_dir, 'feats.pt'))
  # What the harness needs to align atoms: per atom, its token and its name.
  chars = feats['ref_atom_name_chars'][0].argmax(-1).numpy()   # (atoms, 4)
  names = [''.join(chr(c + 32) for c in row).strip() for row in chars]
  a2t = feats['atom_to_token'][0].argmax(-1).numpy()
  pad = feats['atom_pad_mask'][0].numpy() > 0
  np.savez(os.path.join(out_dir, 'layout.npz'),
           atom_name=np.array(names), atom_token=a2t, atom_pad=pad,
           token_pad=feats['token_pad_mask'][0].numpy() > 0,
           res_type=feats['res_type'][0].argmax(-1).numpy(),
           **{'feat.' + k: v[0].numpy() for k, v in feats.items()
              if hasattr(v, 'numpy') and v.dtype != torch.bool
              and v.ndim >= 1})


def forward(model, out_dir, in_npz, out_npz):
  import torch
  from dataclasses import asdict
  from boltz.main import (Boltz2DiffusionParams, PairformerArgsV2,
                          MSAModuleArgs, BoltzSteeringParams)
  from boltz.model.models.boltz2 import Boltz2

  z = np.load(in_npz)
  dev = 'cuda' if torch.cuda.is_available() and not os.environ.get(
      'BOLTZ_CPU') else 'cpu'
  feats = torch.load(os.path.join(out_dir, 'feats.pt'), weights_only=False)
  feats = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in feats.items()}

  # Exactly `boltz predict`'s construction, with use_kernels off (the fused
  # kernels are an implementation choice, not the model) and the MSA not
  # subsampled (random; our port takes the rows in order).
  diffusion = Boltz2DiffusionParams()
  diffusion.step_scale = 1.5
  steer = BoltzSteeringParams()
  steer.fk_steering = steer.physical_guidance_update = False
  model = Boltz2.load_from_checkpoint(
      os.path.join(CACHE, 'boltz2_conf.ckpt'), strict=True,
      predict_args={'recycling_steps': int(z['recycling_steps']),
                    'sampling_steps': 1, 'diffusion_samples': 1,
                    'max_parallel_samples': 1,
                    'write_confidence_summary': True,
                    'write_full_pae': True, 'write_full_pde': True},
      map_location='cpu', diffusion_process_args=asdict(diffusion), ema=False,
      use_kernels=False, pairformer_args=asdict(PairformerArgsV2()),
      msa_args=asdict(MSAModuleArgs(subsample_msa=False,
                                    use_paired_feature=True)),
      steering_args=asdict(steer))
  model.eval().to(dev)

  pos = torch.tensor(z['positions'], dtype=torch.float32, device=dev)[None]
  noisy = torch.tensor(z['positions_noisy'], dtype=torch.float32,
                       device=dev)[None]
  sigma = float(z['noise_level'])
  sm = model.structure_module
  got = {}

  def sample(atom_mask, num_sampling_steps=None, multiplicity=1,
             max_parallel_samples=None, steering_args=None,
             **network_condition_kwargs):
    kw = dict(network_condition_kwargs, multiplicity=1)
    got['denoised'] = sm.preconditioned_network_forward(
        noisy, sigma, network_condition_kwargs=kw)
    return {'sample_atom_coords': pos, 'diff_token_repr': None}
  sm.sample = sample

  # Stage taps: the first call's output of each top-level module, so a
  # disagreement is placed at a stage before anyone reads code.
  taps = {}

  def hook(name):
    def f(_m, _i, o):
      if name in taps:
        return
      o = o if isinstance(o, (tuple, list)) else (o,)
      for k, x in enumerate(o):
        if hasattr(x, 'float'):
          taps['tap.%s.%d' % (name, k)] = x.detach().float().cpu().numpy().copy()
    return f
  for name in ('input_embedder', 's_init', 'z_init_1', 'z_init_2', 'rel_pos',
               'token_bonds', 'token_bonds_type', 'contact_conditioning',
               'template_module', 'msa_module', 'pairformer_module',
               'distogram_module', 'diffusion_conditioning'):
    m = getattr(model, name, None)
    if m is not None:
      m.register_forward_hook(hook(name))
  for name, m in model.input_embedder.named_children():
    m.register_forward_hook(hook('input_embedder.' + name))

  bf16 = str(z['precision']) == 'bf16'
  with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16,
                                       enabled=bf16):
    out = model(feats, recycling_steps=int(z['recycling_steps']),
                num_sampling_steps=1, diffusion_samples=1,
                max_parallel_samples=1)
  out['denoised'] = got['denoised']
  # The vendor_native_parity contract: these names, batch axis squeezed.
  std = {'s_inputs': taps['tap.input_embedder.0'], 's': out['s'],
         'z': out['z'], 'distogram_logits': out['pdistogram'],
         'denoised': out['denoised'], 'pae': out['pae'], 'pde': out['pde'],
         'plddt': out['plddt']}
  np.savez(out_npz, **taps, **{
      k: (v if isinstance(v, np.ndarray) else v.float().cpu().numpy()
          ).reshape(np.shape(v)[1:] if np.shape(v)[0] == 1 else np.shape(v))
      for k, v in std.items()})


if __name__ == '__main__':
  cmd, *rest = sys.argv[1:]
  {'featurise': featurise, 'forward': forward}[cmd](*rest)
