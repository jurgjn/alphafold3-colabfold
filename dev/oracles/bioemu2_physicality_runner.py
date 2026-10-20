"""Stock BioEmu2's physicality check (check_physicality) on given frames.

    python bioemu2_physicality_runner.py CASE.json COORDS.npy OUT.npz

COORDS: (frames, atoms, 3) angstrom in BioEmu2's atom order. Writes, per mode,
`passed`, `reasons` and every stat of check_allatom_physicality.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioemu2_upstream_runner import system_and_msa  # noqa: E402


def main(case_json, coords_npy, out_npz):
  import mdtraj
  from bioemu2 import physicality as PH
  from bioemu2.sampling import check_physicality
  from bioemu2.structure import AllAtomStructure
  from bioemu2.structure.inference_inputs import topology_from_structure
  with open(case_json) as fh:
    system, _ = system_and_msa(json.load(fh))
  st = AllAtomStructure.from_system(system)
  top = topology_from_structure(st, renamed_asym_ids=system.chain_ids)
  xyz = np.load(coords_npy).astype(np.float32) / 10.0
  traj = mdtraj.Trajectory(xyz, top)
  out = {}
  for mode in ('high_quality', 'no_violations', 'high_quality_per_residue'):
    rep = check_physicality(traj, mode, system=system)
    out['%s.passed' % mode] = rep.passed
    out['%s.reasons' % mode] = np.array([r or '' for r in rep.reasons])
  prot = PH.get_protein_atom_indices(traj.topology)
  if len(prot):
    stats = PH.compute_physicality_stats(traj.atom_slice(prot))
    for k, v in stats.items():
      out['stat.' + k] = v
  np.savez(out_npz, **out)


if __name__ == '__main__':
  main(*sys.argv[1:])
