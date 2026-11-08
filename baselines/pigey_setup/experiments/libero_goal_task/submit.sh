#!/bin/bash
# Submit this directory's protocol through the generic suite submitter.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Preserve this entry point's existing default; PIGEY_ROOT always overrides it.
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch3/workspace/udhanuka_umass_edu-pigey}"
exec bash "$HERE/../submit.sh" "$HERE" "$@"
