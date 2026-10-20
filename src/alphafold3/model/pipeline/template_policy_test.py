"""Unit tests for template_policy: the axes, and the mmCIF rewrite each needs."""

from absl.testing import absltest
from alphafold3.model import model_registry
from alphafold3.model.pipeline import template_policy

# Two residues of a template chain: SER 1 with two altlocs (A at 0.4, B at 0.6), then SEP 2.
_ATOM_SITE = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_comp_id
ATOM 1 N N A SER A 1 1 ? 0.0 0.0 0.0 0.4 SER
ATOM 2 N N B SER A 1 1 ? 0.1 0.0 0.0 0.6 SER
HETATM 3 N N . SEP A 1 2 ? 1.0 0.0 0.0 1.0 SEP
HETATM 4 C CA . SEP A 1 2 ? 2.0 0.0 0.0 1.0 SEP
HETATM 5 P P . SEP A 1 2 ? 3.0 0.0 0.0 1.0 SEP
#
"""


class TemplatePolicyTest(absltest.TestCase):

  def test_default_is_af3(self):
    self.assertTrue(template_policy.is_default({}))
    self.assertTrue(template_policy.is_default(None))
    self.assertFalse(template_policy.is_default({'altloc': 'first'}))

  def test_rejects_unknown_axis_or_value(self):
    with self.assertRaises(ValueError):
      template_policy.resolve({'altloc': 'last'})
    with self.assertRaises(ValueError):
      template_policy.resolve({'colour': 'blue'})

  def test_registry_policies_resolve(self):
    for model, policy in model_registry.TEMPLATE_POLICIES.items():
      with self.subTest(model=model):
        template_policy.resolve(policy)

  def test_global_align_identity_and_subrange(self):
    self.assertEqual(template_policy.global_align('ACDE', 'ACDE'), {0: 0, 1: 1, 2: 2, 3: 3})
    # a query that is the middle of the template chain aligns to it, not to its start
    self.assertEqual(template_policy.global_align('CDE', 'ACDEF'), {0: 1, 1: 2, 2: 3})

  def test_of3_policy(self):
    p = template_policy.resolve(model_registry.TEMPLATE_POLICIES['openfold3'])
    self.assertEqual((p['modified'], p['realign'], p['hit_letter'], p['rank']),
                     ('unk_restype', 'whole_chain', 'parent', True))
    self.assertEqual(model_registry.TEMPLATE_POLICIES['openbind0'], model_registry.TEMPLATE_POLICIES['openfold3'])
    # no other model reads templates OpenFold3's way
    for model, policy in model_registry.TEMPLATE_POLICIES.items():
      if model not in ('openfold3', 'openbind0'):
        self.assertFalse(template_policy.resolve(policy)['rank'])
        self.assertNotEqual(template_policy.resolve(policy)['modified'], 'unk_restype')

  def test_rank_orders_by_identity_times_coverage(self):
    from unittest import mock
    from alphafold3.common import folding_input as fi
    query = 'ACDEFGHIKLMNPQRSTVWY'
    full = {i: i for i in range(20)}
    half = {i: i for i in range(10)}
    tiny = {0: 0}                                                     # score 1/400 < 0.1: dropped
    tpls = [fi.Template(mmcif=str(k), query_to_template_map=m) for k, m in enumerate((half, tiny, full))]
    inp = fi.Input(name='t', chains=[fi.ProteinChain(id='A', sequence=query, ptms=[], unpaired_msa='',
                                                     paired_msa='', templates=tpls)], rng_seeds=[1])
    with mock.patch.object(template_policy, 'template_seqres', return_value=(query, '')):
      out = template_policy.apply(inp, dict(rank=True))
    self.assertEqual([t.mmcif for t in out.chains[0].templates], ['2', '0'])

  def test_first_altloc(self):
    out = template_policy.rewrite_mmcif(_ATOM_SITE, first_altloc=True)
    rows = [l for l in out.splitlines() if l.startswith(('ATOM', 'HETATM'))]
    self.assertEqual([r.split()[1] for r in rows], ['1', '3', '4', '5'])   # altloc B dropped
    self.assertEqual(rows[0].split()[4], '.')

  def test_modified_unk_n(self):
    out = template_policy.rewrite_mmcif(_ATOM_SITE, modified_unk_n=True)
    rows = [l.split() for l in out.splitlines() if l.startswith(('ATOM', 'HETATM'))]
    sep = [r for r in rows if r[8] == '2']
    self.assertEqual(len(sep), 1)                                # only its N
    self.assertEqual((sep[0][0], sep[0][3], sep[0][5], sep[0][14]), ('ATOM', 'N', 'UNK', 'UNK'))
    self.assertEqual(len([r for r in rows if r[8] == '1']), 2)   # altlocs untouched

  def test_no_rewrite_is_identity_on_rows(self):
    out = template_policy.rewrite_mmcif(_ATOM_SITE)
    rows = [l.split() for l in out.splitlines() if l.startswith(('ATOM', 'HETATM'))]
    self.assertEqual(len(rows), 5)


if __name__ == '__main__':
  absltest.main()
