# Where the dev shell scripts find things -- the shell side of dev/af3_dev_paths.py.
#
#   . "$(dirname "${BASH_SOURCE[0]}")/<...>/af3_dev_paths.sh"
#
# AF3_DEV_HOME    the root of the home-relative layout the scripts name: vendor
#                 checkouts, venvs, weights, test structures (default: $HOME)
# AF3_PORTED_DIR  converted blobs, <model>/ under it (default: $AF3_DEV_HOME/ported)
# REPO            this repository, wherever it is checked out (from this file)
#
# Every default reproduces the layout the scripts were written on.
: "${AF3_DEV_HOME:=$HOME}"
: "${AF3_PORTED_DIR:=$AF3_DEV_HOME/ported}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export AF3_DEV_HOME AF3_PORTED_DIR REPO
