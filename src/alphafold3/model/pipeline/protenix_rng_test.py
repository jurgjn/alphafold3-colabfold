"""The leaving-group replay against Python's own random.sample, and the bonds native's polymer builder samples at."""

import random

from absl.testing import absltest
from alphafold3.model.pipeline import protenix_rng


def _comp(kind, atoms, bonds):
  """a CCD entry: atoms as (name, element, leaving) and bonds as name pairs"""
  return {'_chem_comp.type': [kind],
          '_chem_comp_atom.atom_id': [a for a, _, _ in atoms],
          '_chem_comp_atom.type_symbol': [e for _, e, _ in atoms],
          '_chem_comp_atom.pdbx_leaving_atom_flag': ['Y' if l else 'N' for _, _, l in atoms],
          '_chem_comp_bond.atom_id_1': [a for a, _ in bonds],
          '_chem_comp_bond.atom_id_2': [b for _, b in bonds]}


_CCD = {
    'AAA': _comp('L-PEPTIDE LINKING', [('N', 'N', 0), ('C', 'C', 0), ('OXT', 'O', 1), ('H2', 'H', 1)],
                 [('N', 'C'), ('C', 'OXT'), ('N', 'H2')]),
    # two leaving groups on C, so random.sample draws below 2
    'BBB': _comp('L-PEPTIDE LINKING', [('N', 'N', 0), ('C', 'C', 0), ('O1', 'O', 1), ('O2', 'O', 1)],
                 [('N', 'C'), ('C', 'O1'), ('C', 'O2')]),
    'CAP': _comp('NON-POLYMER', [('N', 'N', 0)], []),
}


class ProtenixRngTest(absltest.TestCase):

  def test_leaving_group_draws_consume_what_random_sample_does(self):
    for n in range(1, 6):
      for b in range(1, n + 1):
        for seed in range(20):
          want = random.Random(seed)
          want.sample(list(range(n)), b)
          self.assertEqual(protenix_rng.mc_dropout_decision(seed, [(n, b)]),
                           want.random() < protenix_rng.MC_APPLY_RATE)

  def test_draws_follow_the_polymer_bonds(self):
    # C of every residue but the last; N carries only a hydrogen group; no bond into a non-polymer cap
    draws = protenix_rng.leaving_group_draws([['AAA', 'BBB', 'AAA', 'CAP']], _CCD)
    self.assertEqual(draws, [(1, 1), (2, 1)])


if __name__ == '__main__':
  absltest.main()
