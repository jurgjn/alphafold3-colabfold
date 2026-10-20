"""BioEmu2's steered samplers and their potentials, in JAX.

A transcription of bioemu2 @7846769's denoising/steering: the default
physicality potentials (potentials.get_default_potentials) evaluated on the
tables pipeline/bioemu2_steering.py builds, and the three steered samplers its
presets name -- `edm_gd` (gradient guidance on the denoised estimate),
`edm_fk` (the same plus Feynman-Kac resampling) and `edm_fkc` (the Feynman-Kac
corrector, which differentiates the potential through the denoiser).

Every sample is a particle; resampling couples them, so these samplers step
all particles together rather than vmapping one over samples. The vendor
evaluates a clash only for atom pairs within a 10 A cutoff in some particle; a
pair beyond it has zero energy and gradient, so the dense masked sums here are
the same energies.

Every random draw can be injected (`draws`): the initial positions, each
step's rotation, translation and noise, and each resampling's uniform -- how
the parity harness replays stock BioEmu2's.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

# The parity harness's draws for one steered fold ({x0, rot, trans, eps,
# uniform}); None in a run.
INJECT = None

SIGMA_DATA = 16.0
SIGMA_MIN = 0.0004
SIGMA_MAX = 160.0
RHO = 7


# --- parameter schedules (potentials.ParameterSchedule) -------------------------------

def exponential(start, end, alpha):
  return lambda t: (start + (end - start) * (math.exp(alpha * t) - 1) / (math.exp(alpha) - 1)
                    if alpha != 0 else start + (end - start) * t)


def piecewise(thresholds, values):
  def f(t):
    i = 0
    while i < len(thresholds) and t > thresholds[i]:
      i += 1
    return values[i]
  return f


def _const(v):
  return lambda t: v


# get_default_potentials: name -> {parameter: schedule}
DEFAULT = (
    ('SymmetricChainCOMPotential', dict(guidance_interval=4, guidance_weight=_const(0.5),
                                        resampling_weight=_const(0.5),
                                        buffer=exponential(1.0, 5.0, -2.0))),
    ('InterChainVDWOverlapPotential', dict(guidance_interval=5,
                                           guidance_weight=piecewise([0.4], [0.125, 0.0]),
                                           resampling_weight=piecewise([0.6], [0.01, 0.0]),
                                           buffer=_const(0.225))),
    ('PoseBustersPotential', dict(guidance_interval=1, guidance_weight=_const(0.10),
                                  resampling_weight=_const(0.1))),
    ('ChiralAtomPotential', dict(guidance_interval=1, guidance_weight=_const(0.20),
                                 resampling_weight=_const(1.0))),
    ('StereoBondPotential', dict(guidance_interval=1, guidance_weight=_const(0.10),
                                 resampling_weight=_const(1.0))),
    ('PlanarBondPotential', dict(guidance_interval=1, guidance_weight=_const(0.10),
                                 resampling_weight=_const(1.0))),
    ('IntraChainVDWOverlapPotential', dict(guidance_interval=5,
                                           guidance_weight=piecewise([0.4], [0.125, 0.0]),
                                           resampling_weight=piecewise([0.6], [0.01, 0.0]),
                                           buffer=_const(0.225))),
    ('PolymerLocalClashPotential', dict(guidance_interval=1,
                                        guidance_weight=piecewise([0.4], [0.125, 0.0]),
                                        resampling_weight=_const(0.1),
                                        minimum_distance=_const(2.0))),
    ('BondLengthPotential', dict(guidance_interval=1, guidance_weight=_const(0.05),
                                 resampling_weight=_const(0.1))),
    ('BondAnglePotential', dict(guidance_interval=1, guidance_weight=_const(0.05),
                                resampling_weight=_const(0.1))),
)
POSEBUSTERS_BUFFERS = dict(bond_buffer=0.20, angle_buffer=0.20, clash_buffer=0.15)
CHIRAL_BUFFER, STEREO_BUFFER, PLANAR_BUFFER = 0.52360, 0.52360, 0.26180


def host_tables(t):
  """The fixed-parameter halves of `_compute_args`, on the host: PoseBusters'
  buffered bounds, the chiral/stereo/planar intervals, the planar impropers."""
  t = dict(t)
  lo = np.asarray(t['rdkit_lower_bounds'], np.float32).copy()
  hi = np.asarray(t['rdkit_upper_bounds'], np.float32).copy()
  bond = np.asarray(t['rdkit_bounds_bond_mask'], bool)
  ang = np.asarray(t['rdkit_bounds_angle_mask'], bool)
  b = POSEBUSTERS_BUFFERS
  for m, s in ((bond & ~ang, b['bond_buffer']), (~bond & ang, b['angle_buffer']),
               (bond & ang, min(b['bond_buffer'], b['angle_buffer']))):
    lo[m] *= np.float32(1.0 - s)
    hi[m] *= np.float32(1.0 + s)
  m = ~bond & ~ang
  lo[m] *= np.float32(1.0 - b['clash_buffer'])
  hi[m] = np.inf
  t['pb_lower'], t['pb_upper'] = lo, hi
  o = np.asarray(t['chiral_atom_orientations'], bool)
  t['chiral_lower'] = np.where(o, CHIRAL_BUFFER, -np.inf).astype(np.float32)
  t['chiral_upper'] = np.where(o, np.inf, -CHIRAL_BUFFER).astype(np.float32)
  o = np.asarray(t['stereo_bond_orientations'], bool)
  t['stereo_lower'] = np.where(o, np.float32(np.pi - STEREO_BUFFER), -np.inf).astype(np.float32)
  t['stereo_upper'] = np.where(o, np.inf, STEREO_BUFFER).astype(np.float32)
  pb = np.asarray(t['planar_bond_index'], np.int64).reshape(6, -1).T          # (K, 6)
  imp = np.array([[1, 2, 3, 0], [4, 5, 0, 3]]).T                             # (4, 2)
  t['planar_improper_index'] = pb[:, imp].swapaxes(0, 1).reshape(4, -1)       # (4, 2K)
  # SymmetricChainCOMPotential: pairs of chains with more than one atom
  sym = np.asarray(t['symmetric_chain_index'], np.int64).reshape(2, -1)
  big = np.asarray(t['chain_sizes']) > 1
  t['symmetric_chain_pairs'] = sym[:, big[sym[0]] & big[sym[1]]] if sym.size else sym
  return t


# --- geometry -------------------------------------------------------------------------------

def _norm(x):
  """|x| with torch.linalg.norm's zero subgradient at x = 0."""
  s = jnp.sum(x * x, -1)
  return jnp.where(s > 0, jnp.sqrt(jnp.where(s > 0, s, 1.0)), 0.0)


def _flat_bottom(v, lower, upper):
  return jax.nn.relu(lower - v) + jax.nn.relu(v - upper)


def _distance(c, idx):
  return _norm(c[:, idx[0]] - c[:, idx[1]])


def _dihedral(c, idx):
  r_ij = c[:, idx[0]] - c[:, idx[1]]
  r_kj = c[:, idx[2]] - c[:, idx[1]]
  r_kl = c[:, idx[2]] - c[:, idx[3]]
  n_ijk = jnp.cross(r_ij, r_kj)
  n_jkl = jnp.cross(r_kj, r_kl)
  sin = (r_kj * jnp.cross(n_ijk, n_jkl)).sum(-1) / _norm(r_kj)
  cos = (n_ijk * n_jkl).sum(-1)
  return jnp.arctan2(sin, cos)


def _angle(c, idx, eps=1e-6):
  a = c[:, idx[0]] - c[:, idx[2]]
  b = c[:, idx[1]] - c[:, idx[2]]
  a = a / jnp.maximum(_norm(a), eps)[..., None]
  b = b / jnp.maximum(_norm(b), eps)[..., None]
  return jnp.arccos(jnp.clip((a * b).sum(-1), -1.0 + eps, 1.0 - eps))


def _overlap(c, idx, minimum):
  d = jnp.sqrt(jnp.sum(jnp.square(c[:, idx[0]] - c[:, idx[1]]), -1) + 1e-12)
  return jax.nn.relu(minimum - d).sum(-1)


def _dense_overlap(c, eligible, chain, radius, scale, same_chain, block=256):
  """sum over i < j, both eligible, same (or different) chain, of
  relu((r_i + r_j) scale - |x_i - x_j|) -- in row blocks."""
  n = c.shape[1]
  nb = -(-n // block)
  pad = nb * block - n
  cp = jnp.pad(c, ((0, 0), (0, pad), (0, 0)))
  ep = jnp.pad(eligible, (0, pad))
  chp = jnp.pad(chain, (0, pad), constant_values=-1)
  rp = jnp.pad(radius, (0, pad))
  cols = jnp.arange(n)

  def rows(b):
    r = b * block + jnp.arange(block)
    xr = jax.lax.dynamic_slice_in_dim(cp, b * block, block, 1)            # (P, B, 3)
    d = jnp.sqrt(jnp.sum(jnp.square(xr[:, :, None] - c[:, None]), -1) + 1e-12)   # (P, B, n)
    er = jax.lax.dynamic_slice_in_dim(ep, b * block, block)
    cr = jax.lax.dynamic_slice_in_dim(chp, b * block, block)
    rr = jax.lax.dynamic_slice_in_dim(rp, b * block, block)
    sc = (cr[:, None] == chain[None]) if same_chain else (cr[:, None] != chain[None])
    m = er[:, None] & eligible[None] & sc & (r[:, None] < cols[None])
    clash = (rr[:, None] + radius[None]) * scale
    return jnp.where(m[None], jax.nn.relu(clash[None] - d), 0.0).sum((1, 2))
  return jax.lax.map(rows, jnp.arange(nb)).sum(0)


def _excluded_overlap(c, keys, n_pad, eligible, chain, radius, scale, same_chain):
  """The same sum over only the excluded (bonded) pairs, to subtract."""
  if keys.shape[0] == 0:
    return jnp.zeros(c.shape[0])
  i, j = keys // n_pad, keys % n_pad
  sc = (chain[i] == chain[j]) if same_chain else (chain[i] != chain[j])
  m = eligible[i] & eligible[j] & sc & (i < j)
  d = jnp.sqrt(jnp.sum(jnp.square(c[:, i] - c[:, j]), -1) + 1e-12)
  clash = (radius[i] + radius[j]) * scale
  return jnp.where(m[None], jax.nn.relu(clash[None] - d), 0.0).sum(-1)


# --- the potentials ---------------------------------------------------------------------------

def energies(c, t, p):
  """Every default potential's per-particle energy, stacked (10, P), at the
  step's parameters `p` (buffers)."""
  P = c.shape[0]
  zero = jnp.zeros(P)
  out = []
  # 0 SymmetricChainCOMPotential
  sp = t['symmetric_chain_pairs']
  if sp.shape[1]:
    pad = t['atom_pad_mask']
    chain = t['atom_chain_id']
    n_chain = t['chain_sizes'].shape[0]
    w = (pad[:, None] & (chain[:, None] == jnp.arange(n_chain)[None])).astype(c.dtype)   # (N, C)
    com = jnp.einsum('pna,nc->pca', c, w) / jnp.maximum(w.sum(0), 1)[None, :, None]
    d = _norm(com[:, sp[0]] - com[:, sp[1]])
    out.append(jax.nn.relu(p['sym_buffer'] - d).sum(-1))
  else:
    out.append(zero)
  n_pad = t['atom_pad_mask'].shape[0]
  radius, chain = t['atom_vdw_radius'], t['atom_chain_id']
  # 1 InterChainVDWOverlapPotential
  scale = 1.0 - p['inter_buffer']
  e = t['interchain_steric_atom_mask']
  out.append(_dense_overlap(c, e, chain, radius, scale, False)
             - _excluded_overlap(c, t['interchain_bonded_pair_key'], n_pad, e, chain, radius, scale, False))
  # 2 PoseBustersPotential
  idx = t['rdkit_bounds_index']
  out.append(_flat_bottom(_distance(c, idx), t['pb_lower'], t['pb_upper']).sum(-1)
             if idx.shape[1] else zero)
  # 3 ChiralAtomPotential
  idx = t['chiral_atom_index']
  out.append(_flat_bottom(_dihedral(c, idx), t['chiral_lower'], t['chiral_upper']).sum(-1)
             if idx.shape[1] else zero)
  # 4 StereoBondPotential
  idx = t['stereo_bond_index']
  out.append(_flat_bottom(jnp.abs(_dihedral(c, idx)), t['stereo_lower'], t['stereo_upper']).sum(-1)
             if idx.shape[1] else zero)
  # 5 PlanarBondPotential
  idx = t['planar_improper_index']
  out.append(_flat_bottom(jnp.abs(_dihedral(c, idx)), -jnp.inf, PLANAR_BUFFER).sum(-1)
             if idx.shape[1] else zero)
  # 6 IntraChainVDWOverlapPotential
  scale = 1.0 - p['intra_buffer']
  e = t['polymer_steric_atom_mask']
  out.append(_dense_overlap(c, e, chain, radius, scale, True)
             - _excluded_overlap(c, t['polymer_bonded_pair_key'], n_pad, e, chain, radius, scale, True))
  # 7 PolymerLocalClashPotential
  idx = t['polymer_local_clash_index']
  out.append(_overlap(c, idx, p['minimum_distance']) if idx.shape[1] else zero)
  # 8 BondLengthPotential
  idx = t['bond_index']
  out.append(_flat_bottom(_distance(c, idx), t['bond_lower'], t['bond_upper']).sum(-1)
             if idx.shape[1] else zero)
  # 9 BondAnglePotential
  idx = t['angle_index']
  out.append(_flat_bottom(_angle(c, idx), t['angle_lower'], t['angle_upper']).sum(-1)
             if idx.shape[1] else zero)
  return jnp.stack(out)


def step_parameters(t_norm):
  """Every potential's resolved parameters at normalised time t_norm, as arrays."""
  sched = dict(DEFAULT)
  return {
      'guidance_weight': np.array([s['guidance_weight'](t_norm) for _, s in DEFAULT], np.float32),
      'resampling_weight': np.array([s['resampling_weight'](t_norm) for _, s in DEFAULT], np.float32),
      'sym_buffer': np.float32(sched['SymmetricChainCOMPotential']['buffer'](t_norm)),
      'inter_buffer': np.float32(sched['InterChainVDWOverlapPotential']['buffer'](t_norm)),
      'intra_buffer': np.float32(sched['IntraChainVDWOverlapPotential']['buffer'](t_norm)),
      'minimum_distance': np.float32(sched['PolymerLocalClashPotential']['minimum_distance'](t_norm)),
  }


GUIDANCE_INTERVAL = np.array([s['guidance_interval'] for _, s in DEFAULT])


def _buffers(p):
  return {k: p[k] for k in ('sym_buffer', 'inter_buffer', 'intra_buffer', 'minimum_distance')}


# --- shared sampler pieces ------------------------------------------------------------------

def schedule(num_steps):
  """edm_sample_schedule: num_steps Karras levels (x sigma_data), then 0. float32."""
  inv = 1.0 / RHO
  steps = np.arange(num_steps, dtype=np.float32)
  s = (np.float32(SIGMA_MAX) ** inv + steps / np.float32(num_steps - 1)
       * (np.float32(SIGMA_MIN) ** inv - np.float32(SIGMA_MAX) ** inv)) ** RHO
  return np.concatenate([(s * np.float32(SIGMA_DATA)).astype(np.float32), [np.float32(0.0)]])


def _augment(x, mask, rot, trans):
  """augment_coordinates per particle: centre on the masked mean, x @ R + t."""
  w = mask[None, :, None].astype(x.dtype)
  centre = (x * w).sum(1, keepdims=True) / w.sum(1, keepdims=True)
  return jnp.einsum('pnd,pds->pns', x - centre, rot) + trans[:, None, :]


def _rigid_align(x, ref, mask):
  """weighted_rigid_align of x onto ref, per particle (Kabsch, weights = mask)."""
  w = mask.astype(x.dtype)[:, None]

  def one(a, b):
    ac = (a * w).sum(0) / w.sum()
    bc = (b * w).sum(0) / w.sum()
    h = ((a - ac) * w).T @ (b - bc)
    u, _, vt = jnp.linalg.svd(h)
    d = jnp.sign(jnp.linalg.det(u @ vt))
    r = u @ jnp.diag(jnp.array([1.0, 1.0, d], x.dtype)) @ vt
    return (a - ac) @ r + bc
  return jax.vmap(one)(x, ref)


def _random_rotations(key, n):
  """compute_random_augmentation's rotations: normalised Gaussian quaternions."""
  q = jax.random.normal(key, (n, 4))
  q = q / jnp.where(q[:, :1] < 0, -1.0, 1.0) / jnp.linalg.norm(q, axis=-1, keepdims=True)
  r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  two_s = 2.0 / (q * q).sum(-1)
  m = jnp.stack([1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                 two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                 two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)], -1)
  return m.reshape(n, 3, 3)


def grouped_softmax(logw, group):
  g = logw.reshape(-1, group)
  return jax.nn.softmax(g, axis=-1).reshape(-1)


def systematic_resample(w, group, u):
  """systematic_resample with one uniform per group (`u`, shape (groups,))."""
  g = w.reshape(-1, group)
  pos = (jnp.arange(group)[None] + u[:, None]) / group
  idx = jax.vmap(jnp.searchsorted)(jnp.cumsum(g, -1), pos)
  idx = jnp.minimum(idx, group - 1)
  return (idx + jnp.arange(g.shape[0])[:, None] * group).reshape(-1)


def _step_draws(key, inj, k, n, shape):
  """This step's rotation, translation and unit noise: injected, or from the key."""
  if inj is not None:
    return inj['rot'], inj['trans'], inj['eps']
  a, b, c = jax.random.split(jax.random.fold_in(key, k + 1), 3)
  return _random_rotations(a, n), jax.random.normal(b, (n, 3)), jax.random.normal(c, shape)


def _uniform(key, inj, k, groups):
  if inj is not None:
    return inj['uniform']
  return jax.random.uniform(jax.random.fold_in(key, 100000 + k), (groups,))


def _x0(key, injected, shape):
  if injected and 'x0' in injected:
    return jnp.asarray(injected['x0'], jnp.float32)
  return jax.random.normal(jax.random.fold_in(key, 0), shape)


def _per_step_injection(injected, steps, uniform_by):
  """The injected draws as scan inputs: rot/trans/eps per step, and a uniform
  per step (`uniform_by[k]` indexes the injected uniforms)."""
  if not injected:
    return None
  u = np.asarray(injected['uniform'])
  return {'rot': jnp.asarray(injected['rot'][:steps]), 'trans': jnp.asarray(injected['trans'][:steps]),
          'eps': jnp.asarray(injected['eps'][:steps]),
          'uniform': jnp.asarray(u[np.minimum(uniform_by, len(u) - 1)])}


# --- edm_steered_sampler (presets edm_gd, edm_fk) ---------------------------------------------

def steered_sample(denoise, tables, mask, num_particles, key, *, num_steps=200, step_scale=1.0,
                   noise_scale=1.0, gamma_0=0.8, gamma_min=1.0, resampling_interval=3,
                   resampling_group_size=None, fk_lambda=4.0, num_gd_steps=20, companion=None,
                   injected=None, scan=None):
  """edm_steered_sampler. `denoise(x (P, N, 3), sigma, companion) -> x0_hat`.
  resampling_interval None is edm_gd (guidance only). Returns (P, N, 3).

  `scan` is the scan to step with (hk.scan inside a haiku transform). Every
  per-step constant -- the schedule, the potentials' parameters at that step's
  t, whether it resamples -- is computed on the host and scanned over."""
  import haiku as hk
  scan = scan or hk.scan
  t = {k: jnp.asarray(v) for k, v in tables.items()}
  P = num_particles
  group = resampling_group_size or P
  sig = schedule(num_steps)
  gam = np.where(sig > gamma_min, gamma_0, 0.0).astype(np.float32)
  shape = (P,) + mask.shape + (3,)
  last = num_steps - 1
  ks = np.arange(num_steps)
  s_tm, s_t, g = sig[:-1], sig[1:], gam[1:]
  t_hat = (s_tm * (1 + g)).astype(np.float32)
  noise_var = (noise_scale ** 2 * np.maximum(t_hat.astype(np.float64) ** 2 - s_tm.astype(np.float64) ** 2, 0.0))
  resample = (np.zeros(num_steps, bool) if resampling_interval is None
              else (ks % resampling_interval == 0) | (ks == last))
  prm = [step_parameters(1.0 - (k + 1) / num_steps) for k in ks]
  coef = np.zeros((num_steps, max(num_gd_steps, 1), len(DEFAULT)), np.float32)
  for k in ks:
    if k < last:
      for gs in range(num_gd_steps):
        active = (prm[k]['guidance_weight'] != 0) & (gs % GUIDANCE_INTERVAL == 0)
        coef[k, gs] = np.where(active, prm[k]['guidance_weight'], 0.0)
  xs = {'k': ks, 's_t': s_t, 't_hat': t_hat, 'noise_std': np.sqrt(noise_var).astype(np.float32),
        'noise_var': noise_var.astype(np.float32), 'resample': resample,
        'gd': (ks < last) & (num_gd_steps > 0), 'coef': coef,
        'rw': np.stack([p['resampling_weight'] for p in prm]),
        **{b: np.array([p[b] for p in prm], np.float32)
           for b in ('sym_buffer', 'inter_buffer', 'intra_buffer', 'minimum_distance')},
        'uniform_index': np.cumsum(resample) - 1}
  inj = _per_step_injection(injected, num_steps, np.maximum(np.cumsum(resample) - 1, 0))
  if inj is not None:
    xs['inj'] = inj
  gd_on = num_gd_steps > 0

  def body(carry, x_):
    x, sgu, e_prev, has_prev, comp = carry
    k = x_['k']
    rot, trans, eps = _step_draws(key, x_.get('inj'), k, P, shape)
    x = _augment(x, mask, rot, trans)
    if comp is not None:
      comp = _augment(comp, mask, rot, trans)
    sgu = jnp.einsum('pnd,pds->pns', sgu, rot)
    eps = x_['noise_std'] * eps
    noisy = x + eps
    x0 = denoise(noisy, x_['t_hat'], comp)
    buffers = {b: x_[b] for b in ('sym_buffer', 'inter_buffer', 'intra_buffer', 'minimum_distance')}

    def fk_weights(_):
      en = energies(x0, t, buffers)
      energy = (x_['rw'][:, None] * en).sum(0)
      log_g = jnp.where(has_prev, e_prev - energy, -energy)
      ll = jnp.where(gd_on & (x_['noise_var'] > 0),
                     (jnp.square(eps) - jnp.square(eps + sgu)).sum((-1, -2))
                     / (2.0 * jnp.maximum(x_['noise_var'], 1e-30)), 0.0)
      un = ll + fk_lambda * log_g
      un = jnp.where(jnp.isfinite(un), un, -1e4)
      return grouped_softmax(un, group), energy
    w, energy = jax.lax.cond(x_['resample'], fk_weights,
                             lambda _: (jnp.full((P,), 1.0 / group), e_prev), None)

    def guide(_):
      def inner(gs, upd):
        c = x_['coef'][gs]

        def total(cc):
          e = energies(cc, t, buffers)
          return jnp.where(c[:, None] != 0, c[:, None] * e, 0.0).sum()
        return jnp.where(jnp.any(c != 0), upd - jax.grad(total)(x0 + upd), upd)
      upd = jax.lax.fori_loop(0, num_gd_steps, inner, jnp.zeros_like(x0))
      return x0 + upd, upd * (-step_scale * (x_['s_t'] - x_['t_hat']) / jnp.maximum(x_['t_hat'], 1e-12))
    x0, sgu = jax.lax.cond(x_['gd'], guide, lambda _: (x0, sgu), None)

    u = _uniform(key, x_.get('inj'), x_['uniform_index'], P // group)
    idx = jnp.where(x_['resample'], systematic_resample(w, group, u), jnp.arange(P))
    x, noisy, x0, sgu, energy = x[idx], noisy[idx], x0[idx], sgu[idx], energy[idx]
    if comp is not None:
      comp = comp[idx]
    has_prev = has_prev | x_['resample']
    noisy = _rigid_align(noisy, x0, mask)
    x = noisy + step_scale * (x_['s_t'] - x_['t_hat']) * (noisy - x0) / jnp.maximum(x_['t_hat'], 1e-12)
    return (x, sgu, energy, has_prev, comp), None

  x = jnp.float32(sig[0]) * _x0(key, injected, shape)
  carry = (x, jnp.zeros(shape, jnp.float32), jnp.zeros((P,), jnp.float32), jnp.asarray(False),
           companion)
  (x, *_), _ = scan(body, carry, jax.tree_util.tree_map(jnp.asarray, xs))
  return x


# --- edm_fkc_sampler (preset edm_fkc) ---------------------------------------------------------

def _karras_sigma(t):
  lo, hi = SIGMA_MIN ** (1.0 / RHO), SIGMA_MAX ** (1.0 / RHO)
  return SIGMA_DATA * (lo + t * (hi - lo)) ** RHO


def _t_from_sigma(sigma):
  s = max(sigma / SIGMA_DATA, SIGMA_MIN)
  lo, hi = SIGMA_MIN ** (1.0 / RHO), SIGMA_MAX ** (1.0 / RHO)
  return min(max((s ** (1.0 / RHO) - lo) / (hi - lo), 0.0), 1.0)


def _beta(sigma, beta_power):
  t_norm = _t_from_sigma(sigma)
  s_abs = _karras_sigma(t_norm)
  return ((1.0 + 0.1 * (SIGMA_DATA * SIGMA_MIN) ** 2) * (1.0 - t_norm) ** beta_power
          / (1.0 + 0.1 * s_abs ** 2))


def fkc_sample(denoise, tables, mask, num_particles, key, *, num_steps=200, gamma_0=0.8,
               gamma_min=0.0, resampling_interval=None, resampling_group_size=None,
               ess_threshold=0.9, weight_update='guidance', beta_power=4.0, reheat='sigma',
               companion=None, injected=None, scan=None, grad=None):
  """edm_fkc_sampler. `denoise(x, sigma (P,), companion)` must be differentiable
  in x and sigma: weight_update='guidance' takes dU(D(x, sigma))/dx and /dsigma.
  `grad` is the value_and_grad to take it with (hk.value_and_grad in a transform)."""
  import haiku as hk
  scan = scan or hk.scan
  vgrad = grad or hk.value_and_grad
  if weight_update not in ('smc', 'guidance') or reheat not in ('sigma', 'gamma'):
    raise ValueError('fkc: weight_update is smc|guidance, reheat sigma|gamma')
  if weight_update == 'guidance' and reheat != 'sigma':
    raise ValueError("fkc: guidance weights need reheat='sigma'")
  if (resampling_interval is None) == (ess_threshold is None):
    raise ValueError('fkc: exactly one of resampling_interval and ess_threshold')
  t = {k: jnp.asarray(v) for k, v in tables.items()}
  P = num_particles
  group = resampling_group_size or P
  sig = [float(v) for v in schedule(num_steps)]
  nt = len(sig) - 1
  shape = (P,) + mask.shape + (3,)
  rows = []
  for k in range(nt):
    s_c, s_n = sig[k], sig[k + 1]
    s_hat = (sig[k - 1] if k > 0 else s_c) if reheat == 'sigma' else (
        s_c * (1.0 + (gamma_0 if s_n > gamma_min else 0.0)))
    t_c, t_n = max(1.0 - k / nt, 0.0), max(1.0 - (k + 1) / nt, 0.0)
    t_hat = _t_from_sigma(s_hat)
    rows.append(dict(
        k=k, s_c=s_c, s_n=s_n, s_hat=s_hat, var=s_hat ** 2 - s_c ** 2,
        beta_n=_beta(s_n, beta_power), beta_hat=_beta(s_hat, beta_power), beta_c=_beta(s_c, beta_power),
        fixed=(((k + 1) % resampling_interval == 0) if resampling_interval else False) or k == nt - 1,
        last=k == nt - 1, has_next=s_n > 0,
        **{'%s_%s' % (b, w): step_parameters(tt)[b] for w, tt in (('hat', t_hat), ('c', t_c), ('n', t_n))
           for b in ('sym_buffer', 'inter_buffer', 'intra_buffer', 'minimum_distance')}))
  xs = {k: np.array([r[k] for r in rows], np.float32 if isinstance(rows[0][k], float) else None)
        for k in rows[0]}
  inj = _per_step_injection(injected, nt, np.arange(nt))
  if inj is not None:
    xs['inj'] = inj
  guidance = weight_update == 'guidance'
  smc = weight_update == 'smc'

  def total(x0, x_, when):
    return energies(x0, t, {b: x_['%s_%s' % (b, when)] for b in
                            ('sym_buffer', 'inter_buffer', 'intra_buffer', 'minimum_distance')}).sum(0)

  def body(carry, x_):
    x, logw, pot_c, comp = carry
    rot, trans, eps = _step_draws(None if x_.get('inj') is not None else key, x_.get('inj'),
                                  x_['k'], P, shape)
    x = _augment(x, mask, rot, trans)
    if comp is not None:
      comp = _augment(comp, mask, rot, trans)
    reheated = x + jnp.sqrt(x_['var']) * eps
    s_hat = x_['s_hat']
    if guidance:
      def u(xx, ss):
        d = denoise(xx, ss, comp)
        return total(d, x_, 'hat').sum(), d
      (_, den), (gx, gs) = vgrad(u, argnums=(0, 1), has_aux=True)(reheated, jnp.full((P,), s_hat))
    else:
      den = denoise(reheated, jnp.full((P,), s_hat), comp)
    score = (den - reheated) / s_hat ** 2
    pot_hat = jnp.zeros(P)
    if guidance:
      align = ((-x_['beta_hat'] * gx) * score * (x_['s_c'] ** 2 - x_['s_n'] ** 2) / 2.0).reshape(P, -1).sum(1)
      logw = (logw - x_['beta_n'] * total(den, x_, 'n') + x_['beta_c'] * total(den, x_, 'c')
              + align + x_['beta_hat'] * gs * (x_['s_c'] - x_['s_n']))
    elif smc:
      pot_hat = -x_['beta_hat'] * total(den, x_, 'hat')
      logw = logw + pot_hat - pot_c
    nxt = reheated + s_hat * (s_hat - x_['s_n']) * score
    if guidance:
      nxt = nxt - 0.5 * x_['var'] * x_['beta_hat'] * gx
    pot_n = jnp.zeros(P)
    if smc:
      dn = jax.lax.cond(x_['has_next'] > 0, lambda v: denoise(v, jnp.full((P,), x_['s_n']), comp),
                        lambda v: v, nxt)
      pot_n = -x_['beta_n'] * total(dn, x_, 'n')
      logw = logw + pot_n - pot_hat
    w = grouped_softmax(logw, group)
    if ess_threshold is not None:
      g = w.reshape(-1, group)
      do = (jnp.min(1.0 / (group * jnp.square(g).sum(-1))) < ess_threshold) | (x_['last'] > 0)
    else:
      do = x_['fixed'] > 0
    u_ = _uniform(key, x_.get('inj'), x_['k'], P // group)
    idx = jnp.where(do, systematic_resample(w, group, u_), jnp.arange(P))
    nxt = nxt[idx]
    if comp is not None:
      comp = comp[idx]
    logw = jnp.where(do, 0.0, logw)
    if smc:
      pot_c = pot_n[idx]
    return (nxt, logw, pot_c, comp), None

  x = jnp.float32(sig[0]) * _x0(key, injected, shape)
  carry = (x, jnp.zeros(P), jnp.zeros(P), companion)
  (x, *_), _ = scan(body, carry, jax.tree_util.tree_map(jnp.asarray, xs))
  return x
