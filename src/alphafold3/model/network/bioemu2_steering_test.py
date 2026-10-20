"""BioEmu2 steering: the inputs BioEmu2 refuses are refused here too."""

import os
import tempfile

from absl.testing import absltest
from alphafold3.model.network import bioemu2_steering as steering
from alphafold3.model.pipeline import bioemu2_features
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


if __name__ == '__main__':
  absltest.main()
