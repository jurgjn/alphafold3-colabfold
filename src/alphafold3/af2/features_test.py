"""AF2's protein-only refusal holds on the path run_alphafold.py takes (a batch)."""

import json

from absl.testing import absltest
from alphafold3.af2 import features as af2_features
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import feat_batch

_SEQ = 'GSWKELAAKLAKEAAEKGDPEVAKRALE'


def _protein(chain_id, **kw):
  return {'protein': {'id': chain_id, 'sequence': _SEQ, 'modifications': [],
                      'unpairedMsa': '', 'pairedMsa': '', 'templates': [], **kw}}


def _batch(*sequences):
  fold_input = folding_input.Input.from_json(json.dumps({
      'name': 'af2_scope', 'modelSeeds': [1], 'dialect': 'alphafold3',
      'version': 3, 'sequences': list(sequences)}))
  data = featurisation.featurise_input(
      fold_input, ccd=chemical_components.Ccd(), buckets=None)[0]
  return feat_batch.Batch.from_data_dict(data)


class FromAf3BatchScopeTest(absltest.TestCase):

  def test_protein_is_featurised(self):
    inputs, seq = af2_features.from_af3_batch(_batch(_protein('A'), _protein('B')))
    self.assertEqual(seq, _SEQ * 2)
    self.assertLen(inputs['residue_index'], 2 * len(_SEQ))

  def test_non_protein_chain_is_refused(self):
    for other in ({'rna': {'id': 'B', 'sequence': 'GCGGAU', 'unpairedMsa': ''}},
                  {'dna': {'id': 'B', 'sequence': 'GCGATC'}},
                  {'ligand': {'id': 'B', 'ccdCodes': ['ATP']}}):
      with self.subTest(next(iter(other))):
        with self.assertRaisesRegex(ValueError, 'protein-only'):
          af2_features.from_af3_batch(_batch(_protein('A'), other))

  def test_modified_residue_is_refused(self):
    batch = _batch(_protein('A', modifications=[{'ptmType': 'SEP', 'ptmPosition': 3}]))
    with self.assertRaisesRegex(ValueError, 'modified residue'):
      af2_features.from_af3_batch(batch)


if __name__ == '__main__':
  absltest.main()
