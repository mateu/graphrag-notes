#!/bin/sh
# Keep shell tracing away from credentials; Python never renders child output.
set -eu
exec python3 "$(dirname "$0")/deploy_fam.py" "$@"
