"""chai-1's ESM2 3B tower: ours against the one chai-lab runs.

chai embeds every protein chain alone with a TRACED fp16 ESM2 3B
(chai_lab/data/dataset/embeddings/esm.py, traced_sdpa_esm2_t36_3B_UR50D_fp16.pt):
`<cls>seq<eos>`, its last hidden state, BOS/EOS stripped. Ours is
alphafold3.model.esm.embed(family='esm2') on the converted weights.

Two rows per sequence:
  fp32   the trace with its weights cast to fp32 (`.float()`), on the CPU --
         the model, graded strictly (PARITY <= 1e-3).
  stock  the trace as chai runs it (fp16, on the GPU when there is one) --
         chai's own rounding; INFO, the yardstick for a half-precision tower.

  PYTHONPATH=src:. python dev/oracles/esm2_tower_parity.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.expanduser(os.environ.get(
    'CHAI_ESM_TRACE', '~/chai1_weights/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt'))
VENDOR_PY = os.path.expanduser(os.environ.get('CHAI_VENDOR_PY', '~/venv_chai/bin/python'))
ESM2_DIR = os.path.expanduser(os.environ.get('ESM2_DIR', '~/ported/esm2'))


def vendor(seqs, out):
  """(run in chai's venv) chai's embedding, per sequence, at stock and at fp32."""
  import torch
  from chai_lab.data.dataset.embeddings.esm import esm_tokenizer
  m = torch.jit.load(TRACE, map_location='cpu').eval()
  res = {}
  with torch.no_grad():
    if torch.cuda.is_available():
      g = m.to('cuda:0')
      for i, s in enumerate(seqs):
        ids = torch.asarray(esm_tokenizer.tokenize('<cls>%s<eos>' % s))[None].to('cuda:0')
        res['stock.%d' % i] = g(tokens=ids)[0, 1:-1].float().cpu().numpy()
      m = g.to('cpu')
    # the trace casts to fp16 inside its graph (ScalarType 5): rewritten to
    # fp32 as chai1_upstream_runner does chai's bf16 casts, then fp32 weights
    sys.path.insert(0, HERE)
    from chai1_upstream_runner import _fp32_graphs
    _fp32_graphs(m, from_dtypes=(5, 15))
    m = m.float()
    if os.environ.get('ESM2_EXACT_ROPE'):
      # the trace stores rot_emb.inv_freq as fp16 BUFFERS, so `.float()` keeps
      # fp16-rounded frequencies; this restores ESM2's exact 1/10000^(2i/d)
      for name, b in m.named_buffers():
        if name.endswith('inv_freq'):
          d = 2 * b.numel()
          b.copy_(1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d)))
    for i, s in enumerate(seqs):
      ids = torch.asarray(esm_tokenizer.tokenize('<cls>%s<eos>' % s))[None]
      res['fp32.%d' % i] = m(tokens=ids)[0, 1:-1].float().numpy()
  np.savez(out, **res)


def main():
  sys.path[:0] = [os.path.join(HERE, '..', '..', 'src'), HERE]
  import af3_native_parity as P
  from alphafold3.model import esm
  seqs = {'6mrr': P.P6MRR, 'ubq': P.UBQ, 'long_chain': P._long_chain(),
          'with_x': P.P6MRR[:10] + 'X' + P.P6MRR[11:]}
  names = list(seqs)
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'v.npz')
    with open(os.path.join(tmp, 'seqs.json'), 'w') as f:
      json.dump([seqs[n] for n in names], f)
    r = subprocess.run([VENDOR_PY, '-I', os.path.abspath(__file__), '--vendor',
                        os.path.join(tmp, 'seqs.json'), out], capture_output=True, text=True)
    if r.returncode:
      raise RuntimeError('vendor failed:\n' + r.stderr[-3000:])
    v = dict(np.load(out))
  failed, ours_by = [], {}
  for i, n in enumerate(names):
    ours = esm.embed([seqs[n]], model_dir=ESM2_DIR, family='esm2')
    ours_by[n] = ours
    for tag in ('fp32', 'stock'):
      k = '%s.%d' % (tag, i)
      if k not in v:
        continue
      c, ratio = P.compare(ours, v[k])
      g = P.grade(c, ratio) if tag == 'fp32' else 'INFO'
      print('  %-10s %-5s %-7s corr %.7f max|d|/rms %.2e' % (n, tag, g, c, ratio))
      if g not in ('PARITY', 'CLOSE', 'INFO'):
        failed.append('%s.%s' % (n, tag))
  if os.environ.get('ESM2_FP64'):
    # Ground truth: OUR tower in float64 (the trace cannot run in fp64: it
    # carries fp32 tensor constants). Each fp32 implementation's error against
    # it says whether a residual between the two is fp32 rounding (both alike)
    # or a difference in the function (the trace far, ours near).
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    class _J64:
      float32 = jnp.float64

      def __getattr__(self, k):
        return getattr(jnp, k)
    esm.jnp, esm.COMPUTE_DTYPE = _J64(), jnp.float64

    def attend64(q, k, v, pos, heads, head_dim, half_inv_freq=False):
      # plain softmax attention (the shared kernel has no fp64 CPU path)
      n_len = q.shape[0]
      q = esm._rope(q.reshape(n_len, heads, head_dim), pos, half_inv_freq=half_inv_freq)
      k = esm._rope(k.reshape(n_len, heads, head_dim), pos, half_inv_freq=half_inv_freq)
      v = v.reshape(n_len, heads, head_dim)
      a = jax.nn.softmax(jnp.einsum('ihd,jhd->hij', q, k) * head_dim ** -0.5, axis=-1)
      return jnp.einsum('hij,jhd->ihd', a, v).reshape(n_len, -1)
    esm._attend = attend64
    for i, n in enumerate(names):
      p, dims = esm.load(ESM2_DIR, 'esm2', None)
      ids = esm.lm_input_ids(esm.sequence_ids(seqs[n], 'esm2'))
      ref = np.asarray(esm.forward(ids, p, dims), np.float64)[1:-1]
      rms = np.sqrt((ref ** 2).mean())
      e_o = np.abs(ours_by[n] - ref).max() / rms
      e_v = np.abs(v['fp32.%d' % i] - ref).max() / rms
      print('  %-10s fp64  INFO    err vs fp64 (ours) max|d|/rms: ours fp32 %.2e  trace fp32 %.2e'
            % (n, e_o, e_v))
  print('__VERDICT %s %s' % ('FAIL' if failed else 'PASS', ' '.join(failed)))
  return 1 if failed else 0


if __name__ == '__main__':
  if sys.argv[1:2] == ['--vendor']:
    with open(sys.argv[2]) as f:
      vendor(json.load(f), sys.argv[3])
  else:
    sys.exit(main())
