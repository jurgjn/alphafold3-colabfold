"""Where the dev tools find things: this repository, vendor checkouts, venvs and weights.

The parity harness, benchmarks and probes under dev/ (and scripts/) drive other
projects' code -- vendor checkouts, their virtualenvs, their checkpoints -- and
our own converted blobs. They name those places in the form they were written
for, a home-relative layout:

    ~/venv_boltz/bin/python      a vendor's virtualenv
    ~/protenix                   a vendor checkout
    ~/boltz2_weights/...         a vendor checkpoint
    ~/ported/<model>             our converted blob (converters/ writes it there)
    ~/6MRR.pdb                   a test structure

and resolve them here, at run time, through `dev_path`:

    from af3_dev_paths import dev_path, REPO
    python = dev_path('~/venv_boltz/bin/python')

so a machine with a different layout sets environment variables instead of
editing scripts:

    AF3_DEV_HOME    the root of that layout (default: $HOME)
    AF3_PORTED_DIR  converted blobs, ~/ported/<model> (default: $AF3_DEV_HOME/ported)

This repository itself is never looked for in the layout: `REPO` is where this
file is, so a second checkout (a worktree) runs its own code, not the one at
~/alphafold3. Every default reproduces the layout the scripts were written on,
so with no variables set nothing changes.

Standard library only, and Python 3.8+: vendor runners import it from their own
interpreters (some run with -I), so it is imported by file location --
`sys.path.insert(0, <repo>/dev)` -- not as a package.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_HOME = os.path.abspath(os.path.expanduser(os.environ.get('AF3_DEV_HOME') or '~'))
PORTED_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('AF3_PORTED_DIR') or os.path.join(DEV_HOME, 'ported')))

# A layout path: `~/` (or the absolute home it was written as) at the start of the
# string or after a separator -- never after another path character, so a haiku
# parameter name like 'diffuser/~/transformer/' is left alone.
_LAYOUT = re.compile(r'(?<![\w./~-])(?:~|/home/ubuntu)(?=/|$)(/[^\s:\'"]*)?')


def _resolve(rel):
  """`rel` is what followed `~` ('' or '/...'); returns its place on this machine."""
  rest = rel.lstrip('/')
  head = rest.split('/', 1)[0]
  if head == 'alphafold3':                 # this repository, wherever it is checked out
    return REPO + rest[len('alphafold3'):]
  if head == 'ported':                     # converted blobs
    return PORTED_DIR + rest[len('ported'):]
  return os.path.join(DEV_HOME, rest) if rest else DEV_HOME + rel


def dev_path(s):
  """Rewrite every layout path inside `s` (a path, a PYTHONPATH list, a command line)."""
  return _LAYOUT.sub(lambda m: _resolve(m.group(1) or ''), s)
