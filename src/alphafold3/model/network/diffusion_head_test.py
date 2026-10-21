"""The sampler's guidance hook (diffusion_head.DENOISED_HOOK / make_denoising_body)."""

import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from alphafold3.model.network import diffusion_head

_SHAPE = (4, 3, 3)   # tokens, atoms per token, xyz


def _run(hook, chai=False, steps=3):
  """Scan the step body over a short schedule with a denoiser that predicts the origin."""
  config = diffusion_head.SampleConfig(steps=steps, gamma_0=0.0)
  denoise = lambda x, t, **kw: jnp.zeros_like(x)
  body = diffusion_head.make_denoising_body(
      denoise, jnp.ones(_SHAPE[:2], bool), config, None, chai, hook=hook)
  levels = jnp.asarray([10.0, 5.0, 2.0, 1.0][:steps + 1])
  xs = (levels[1:], levels[:-1])
  if hook is not None:
    xs = (*xs, jnp.arange(steps))
  carry = (jax.random.PRNGKey(0), jnp.ones(_SHAPE))
  _, (_, denoised) = jax.lax.scan(body, carry, xs)
  return np.asarray(denoised)


class DenoisedHookTest(absltest.TestCase):

  def test_hook_replaces_every_denoised_prediction(self):
    hook = lambda x, sigma, step: x + 1.0 + step
    denoised = _run(hook)
    for step in range(3):
      np.testing.assert_allclose(denoised[step], 1.0 + step)

  def test_hook_sees_noise_level_prev(self):
    hook = lambda x, sigma, step: jnp.full_like(x, sigma)
    denoised = _run(hook)
    np.testing.assert_allclose(denoised[:, 0, 0, 0], [10.0, 5.0, 2.0])

  def test_no_hook_is_the_plain_denoiser(self):
    denoised = _run(None)
    np.testing.assert_allclose(denoised, 0.0)

  def test_chai_second_order_step_is_hooked_too(self):
    calls = []

    def hook(x, sigma, step):
      calls.append(step)
      return x
    _run(hook, chai=True)
    # traced once by scan: the first denoising call and the second-order one
    self.assertLen(calls, 2)


if __name__ == '__main__':
  absltest.main()
