"""BioEmu2 featurisation corner cases, each fixed against the stock vendor."""

import json

from absl.testing import absltest
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import bioemu2_features as bf


def _input(sequences, bonds=None):
  j = {'name': 't', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 2,
       'sequences': sequences}
  if bonds:
    j['bondedAtomPairs'] = bonds
  return folding_input.Input.from_json(json.dumps(j))


class BioEmu2FeaturesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ccd = chemical_components.Ccd()

  def test_deuterium_is_hydrogen(self):
    t = bf.ccd_template(self.ccd, 'DOD')
    self.assertEqual(t.names, ['O'])

  def test_glycan_leaving_atoms_per_entity(self):
    # a chitobiose written as two single-NAG chains: one entity, and C's C1
    # donates, so O1 goes from B as well (all_atom_structure's asym_set)
    fi = _input([{'protein': {'id': 'A', 'sequence': 'GSGS', 'unpairedMsa': '>q\nGSGS\n',
                              'pairedMsa': '', 'templates': []}},
                 {'ligand': {'id': 'B', 'ccdCodes': ['NAG']}},
                 {'ligand': {'id': 'C', 'ccdCodes': ['NAG']}}],
                [[['B', 1, 'O4'], ['C', 1, 'C1']]])
    _, keys = bf.featurise(fi, self.ccd)
    names = [(c, a) for c, _, a in keys if c in 'BC']
    self.assertNotIn(('B', 'O1'), names)
    self.assertNotIn(('C', 'O1'), names)
    self.assertIn(('B', 'O4'), names)


if __name__ == '__main__':
  absltest.main()
