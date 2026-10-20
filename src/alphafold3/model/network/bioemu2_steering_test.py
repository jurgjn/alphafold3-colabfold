"""BioEmu2 steering: the inputs BioEmu2 refuses are refused here too."""

import json
import os
import tempfile

from absl.testing import absltest
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model.network import bioemu2_steering as steering
from alphafold3.model.pipeline import bioemu2_features
from alphafold3.model.pipeline import bioemu2_steering as steering_tables
import jax
import jax.numpy as jnp
import numpy as np


def _yaml(text):
  fd, path = tempfile.mkstemp(suffix='.yaml')
  with os.fdopen(fd, 'w') as fh:
    fh.write(text)
  return path


class BioEmu2SteeringTest(absltest.TestCase):

  def test_empty_potentials_refused(self):
    for text in ('potentials: []\n', 'other: 1\n'):
      with self.assertRaises(ValueError):
        steering.load_potentials(_yaml(text))

  def test_schedule_needs_two_steps(self):
    with self.assertRaises(ValueError):
      steering.schedule(1)
    self.assertLen(steering.schedule(2), 3)

  def test_guidance_interval_schedule(self):
    specs = (('PolymerLocalClashPotential', {'guidance_interval': lambda t: 2 if t > 0.5 else 3}, {}),)
    np.testing.assert_array_equal(steering.guidance_interval(0.9, specs), [2])
    np.testing.assert_array_equal(steering.guidance_interval(0.1, specs), [3])

  def test_propagation_needs_a_step(self):
    with self.assertRaises(ValueError):
      bioemu2_features.add_propagation({}, np.zeros((1, 3), np.float32), 1.0, steps=0)

  def test_fkc_reports_the_first_non_finite_step(self):
    fi = folding_input.Input.from_json(json.dumps({
        'name': 't', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 2,
        'sequences': [{'protein': {'id': ['A', 'B'], 'sequence': 'GAG', 'unpairedMsa': '>q\nGAG\n',
                                   'pairedMsa': '', 'templates': []}}]}))
    ccd = chemical_components.Ccd()
    _, _, topo = bioemu2_features.featurise(fi, ccd, return_topology=True)
    t = steering.host_tables(steering_tables.tables(topo, ccd))
    t = {k: v for k, v in t.items() if k != 'graph_bonds'}
    mask = jnp.asarray(np.arange(topo['n_pad']) < len(topo['names']))
    for cut, want in ((None, -1), (5.0, 8)):
      def den(x, s, comp, cut=cut):
        del comp
        return x * 0.5 if cut is None else jnp.where((s < cut)[:, None, None], jnp.nan, x * 0.5)
      _, bad = steering.fkc_sample(den, t, mask, 2, jax.random.PRNGKey(0), num_steps=10,
                                   scan=jax.lax.scan, grad=jax.value_and_grad)
      self.assertEqual(int(bad), want)


if __name__ == '__main__':
  absltest.main()
