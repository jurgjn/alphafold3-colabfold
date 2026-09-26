"""Tests for the triangle multiplication's channel-blocked path."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3.model import model_config
from alphafold3.model.network import modules


class BlockedTriangleMultiplicationTest(parameterized.TestCase):
  """Blocking by channels must keep the parameter names and the output of the stock path."""

  def _run(self, equation, blocked):
    old = modules.TRIMUL_BLOCK_BYTES
    modules.TRIMUL_BLOCK_BYTES = 0 if blocked else 2 ** 62
    try:
      cfg = modules.TriangleMultiplication.Config(
          equation=equation, use_glu_kernel=False, channel_block=16)
      gc = model_config.GlobalConfig()
      f = hk.transform(lambda x, m: modules.TriangleMultiplication(cfg, gc, name='tm')(x, m))
      x = jax.random.normal(jax.random.PRNGKey(0), (40, 40, 64))
      m = (jax.random.uniform(jax.random.PRNGKey(1), (40, 40)) > 0.1).astype(jnp.float32)
      params = f.init(jax.random.PRNGKey(2), x, m)
      params = jax.tree.map(
          lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape, p.dtype) * 0.2, params)
      with jax.default_matmul_precision('highest'):
        return jax.tree.map(jnp.shape, params), np.asarray(f.apply(params, None, x, m))
    finally:
      modules.TRIMUL_BLOCK_BYTES = old

  @parameterized.parameters('ikc,jkc->ijc', 'kjc,kic->ijc')
  def test_matches_stock(self, equation):
    stock_params, stock = self._run(equation, blocked=False)
    blocked_params, blocked = self._run(equation, blocked=True)
    self.assertEqual(stock_params, blocked_params)
    np.testing.assert_allclose(blocked, stock, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
