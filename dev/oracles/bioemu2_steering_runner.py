"""Stock BioEmu2's steering potentials, dumped for a by-hand comparison (no harness drives this runner).

    python bioemu2_steering_runner.py CASE.json COORDS.npy OUT.npz

Runs in ~/venv_bioemu2. Builds the System as bioemu2_upstream_runner does,
prepares every potential of get_default_potentials(), and writes for each
(`<i>.<Name>.*`): every array it prepared (its index tables and bounds), and
its energy and gradient on COORDS (particles, atoms, 3, angstrom, BioEmu2's
atom order) at t = 1, 0.5 and 0.1. Also the bonded geometry graph
(`graph.bonds`) and every potential's parameter values at those t.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioemu2_upstream_runner import system_and_msa  # noqa: E402

TS = (1.0, 0.5, 0.1)


def main(case_json, coords_npy, out_npz):
  import torch
  from bioemu2.denoising.steering.geometry_topology import build_bonded_geometry_graph
  from bioemu2.denoising.steering.potentials import get_default_potentials
  from bioemu2.structure import AllAtomStructure
  with open(case_json) as fh:
    j = json.load(fh)
  system, _ = system_and_msa(j)
  coords = torch.tensor(np.load(coords_npy), dtype=torch.float32)
  out = {}
  graph = build_bonded_geometry_graph(AllAtomStructure.from_system(system, mode='structure'))
  out['graph.bonds'] = np.array(graph.bonds, np.int64).reshape(-1, 2)
  out['graph.angles'] = np.array(graph.angles, np.int64).reshape(-1, 3)
  for i, p in enumerate(get_default_potentials()):
    name = '%d.%s' % (i, type(p).__name__)
    p.prepare(system)
    for k, v in p.feats.items():
      if torch.is_tensor(v) and k not in ('coords', 'coords_cond'):
        out['%s.feat.%s' % (name, k)] = v.cpu().numpy()
    for t in TS:
      params = p.compute_parameters(t) or {}
      for k, v in params.items():
        out['%s.param.%s.%g' % (name, k, t)] = np.float64(v)
      c = coords.clone().requires_grad_(True)
      e = p(c, t=t)
      g = torch.autograd.grad(e.sum(), c, allow_unused=True)[0] if e.requires_grad else None
      out['%s.energy.%g' % (name, t)] = e.detach().numpy()
      out['%s.grad.%g' % (name, t)] = (g if g is not None else torch.zeros_like(c)).numpy()
  np.savez(out_npz, **out)


if __name__ == '__main__':
  main(*sys.argv[1:])
