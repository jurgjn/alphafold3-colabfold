"""Philox against known answers, and the in-graph Philox and dropout mask against the host replay, bit for bit."""

import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from alphafold3.model.network import torch_philox
from alphafold3.model.pipeline import protenix_rng


def _keep_reference(seed, offset, n, c, keep, grid, vec=8):
  """torch's fused dropout kernel, element by element: thread vec-chunks, a curand4 per 4 elements"""
  e = np.arange(n * n * c)
  stride = grid * 256 * vec
  it, within = e // stride, e % stride
  t, j = within // vec, within % vec
  u = protenix_rng.philox(seed, t, offset // 4 + it * (vec // 4) + j // 4)
  u = np.take_along_axis(u, (j % 4)[:, None], -1)[:, 0]
  f = u.astype(np.float32) * np.float32(2.3283064e-10) + np.float32(2.3283064e-10 / 2.0)
  return (f < np.float32(keep)).reshape(n, n, c)


class TorchPhiloxTest(absltest.TestCase):

  def test_known_answers(self):
    # Random123's philox4x32-10 vectors: counter (c0..c3), key (k0, k1) -> output, on numpy and jax arrays
    for ctr, key, want in (
        ((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
        ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
        ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
         (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))):
      for xp in (np, jnp):
        u32 = lambda v: xp.asarray([v], dtype=np.uint32)
        with np.errstate(over='ignore'):
          got = torch_philox.philox4x32(tuple(map(u32, key)), tuple(map(u32, ctr)))
        self.assertEqual(tuple(int(np.asarray(x)[0]) for x in got), want)

  def test_philox_matches_numpy(self):
    rng = np.random.default_rng(0)
    sub = rng.integers(0, 2 ** 20, 64).astype(np.uint32)
    blk = rng.integers(0, 2 ** 20, 64).astype(np.uint32)
    got = np.stack(torch_philox.philox(42, jnp.asarray(sub), jnp.asarray(blk)), -1)
    np.testing.assert_array_equal(np.asarray(got), protenix_rng.philox(42, sub, blk))

  def test_dropout_mask_matches_kernel_mapping(self):
    n, c, grid = 10, 16, 3                         # a small grid so the grid-stride loop wraps
    want = _keep_reference(7, 24, n, c, 0.6, grid)
    got = np.asarray(torch_philox.dropout_keep(7, 24, n, (12, 12, c), keep=0.6, grid=grid))
    np.testing.assert_array_equal(got[:n, :n], want)
    self.assertFalse(got[n:].any() or got[:, n:].any())


if __name__ == '__main__':
  absltest.main()
