"""cuRAND Philox4_32_10, and the keep mask of torch's fused dropout kernel on a (n, n, c) tensor.

philox4x32 runs on uint32 numpy or jax arrays alike: pipeline.protenix_rng replays native's draws with it on the host,
dropout_keep builds the mask in-graph, where it is too large to ship as a feature. uint32 only, so the 64-bit
products are assembled from 16-bit limbs.
"""
import jax.numpy as jnp
import numpy as np

_MA, _MB = 0xD2511F53, 0xCD9E8D57
_WA, _WB = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
_LO16 = np.uint32(0xFFFF)


def _mulhilo(a, b):
  """(high, low) 32 bits of a * b for uint32 a and a Python-int constant b"""
  bl, bh = np.uint32(b & 0xFFFF), np.uint32(b >> 16)
  al, ah = a & _LO16, a >> 16
  ll, lh, hl, hh = al * bl, al * bh, ah * bl, ah * bh
  cross = (ll >> 16) + (lh & _LO16) + (hl & _LO16)
  hi = hh + (lh >> 16) + (hl >> 16) + (cross >> 16)
  return hi, a * np.uint32(b)


def philox4x32(key, counter):
  """The ten rounds: key (k0, k1) and counter (c0, c1, c2, c3) as uint32 arrays -> 4 uint32 outputs"""
  (k0, k1), (c0, c1, c2, c3) = key, counter
  for r in range(10):
    hi0, lo0 = _mulhilo(c0, _MA)
    hi1, lo1 = _mulhilo(c2, _MB)
    c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    if r < 9:
      k0, k1 = k0 + _WA, k1 + _WB
  return c0, c1, c2, c3


def philox(seed, subseq, block):
  """4 uint32 outputs for key=seed (below 2**32; may be traced) and counter=(block, subseq), both uint32 arrays"""
  block, subseq = block.astype(jnp.uint32), subseq.astype(jnp.uint32)
  zero = jnp.zeros((), jnp.uint32)
  return philox4x32((jnp.asarray(seed).astype(jnp.uint32), zero), (block, zero, subseq, zero))


def dropout_keep(seed, offset, n_real, shape, *, keep, grid, vec=8):
  """keep mask torch's fused dropout draws for a contiguous (n_real, n_real, c) tensor, laid into `shape` (bucket-padded)

  grid is the launch grid native uses on this device (protenix_rng.grid_size); vec is torch's vector width for the
  dtype native runs in (8 for bfloat16). The padding outside n_real is False.
  """
  nb, _, c = shape
  assert c % 4 == 0 and vec % 4 == 0
  i = jnp.arange(nb, dtype=jnp.uint32)[:, None, None]
  j = jnp.arange(nb, dtype=jnp.uint32)[None, :, None]
  q = jnp.arange(c // 4, dtype=jnp.uint32)[None, None, :]
  n = jnp.asarray(n_real, jnp.uint32)
  e = (i * n + j) * jnp.uint32(c) + q * jnp.uint32(4)         # first element of the quad; its 4 draws are one curand4
  stride = jnp.asarray(grid).astype(jnp.uint32) * jnp.uint32(256 * vec)
  it, within = e // stride, e % stride
  t = within // jnp.uint32(vec)
  draw = it * jnp.uint32(vec // 4) + (within % jnp.uint32(vec)) // jnp.uint32(4)
  outs = philox(seed, t, jnp.asarray(offset, jnp.uint32) // jnp.uint32(4) + draw)
  u = jnp.stack(outs, -1).reshape(nb, nb, c)
  f = u.astype(jnp.float32) * jnp.float32(2.3283064e-10) + jnp.float32(2.3283064e-10 / 2.0)
  valid = (jnp.arange(nb)[:, None] < n_real) & (jnp.arange(nb)[None, :] < n_real)
  return (f < jnp.float32(keep)) & valid[..., None]
