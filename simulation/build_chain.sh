#!/usr/bin/env bash
# Fetch dependencies and build the chain binary the simulation scripts drive.
#
# The binary is a Cosmos-SDK application exposing the x/adaptivecommittee module:
# it stores the controller state, applies the mixed-weight distribution
# q_i(lambda) = (1-lambda)*w_i + lambda*b_i, draws committees and emits the
# per-draw diagnostic event the runners read.
#
#   bash simulation/build_chain.sh
#
# On success the binary lands in simulation/chain/build/ and the path to export
# as POC_CHAIND is printed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN="$HERE/chain"
BINARY="chain-five-threed"
OUT="$CHAIN/build/$BINARY"

REQUIRED_GO="1.24"

say()  { printf '\033[0;32m[build]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build] warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[0;31m[build] error:\033[0m %s\n' "$*" >&2; exit 1; }

# ── Go toolchain ──────────────────────────────────────────────────────────────
# Go is often installed outside PATH; look where it usually lands before giving up.
if ! command -v go >/dev/null 2>&1; then
  for candidate in "$HOME/.local/go/bin" /usr/local/go/bin /opt/go/bin; do
    if [[ -x "$candidate/go" ]]; then
      export PATH="$candidate:$PATH"
      say "found Go in $candidate"
      break
    fi
  done
fi

command -v go >/dev/null 2>&1 || die "Go not found.
  Install Go $REQUIRED_GO or newer from https://go.dev/dl/ and re-run.
  If it is already installed outside PATH, export it first:
    export PATH=\"\$HOME/.local/go/bin:\$PATH\""

GO_VERSION="$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
say "Go $GO_VERSION"

# Compare only major.minor; go.mod pins the patch level itself.
go_major="${GO_VERSION%%.*}"
go_rest="${GO_VERSION#*.}"
go_minor="${go_rest%%.*}"
req_major="${REQUIRED_GO%%.*}"
req_minor="${REQUIRED_GO#*.}"
if (( go_major < req_major )) || { (( go_major == req_major )) && (( go_minor < req_minor )); }; then
  die "Go $REQUIRED_GO or newer is required, found $GO_VERSION."
fi

[[ -f "$CHAIN/go.mod" ]] || die "chain sources not found at $CHAIN"

# ── Dependencies ──────────────────────────────────────────────────────────────
cd "$CHAIN"
say "downloading modules (first run fetches the Cosmos SDK, this takes a while)"
go mod download

say "verifying module checksums against go.sum"
go mod verify >/dev/null || warn "go mod verify reported a mismatch; inspect before trusting the build"

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p "$CHAIN/build"
say "building $BINARY"
go build -o "$OUT" ./cmd/$BINARY

[[ -x "$OUT" ]] || die "build finished but $OUT is missing"

# ── Smoke check ───────────────────────────────────────────────────────────────
if ! "$OUT" tx adaptivecommittee --help >/dev/null 2>&1; then
  die "the binary does not expose the adaptivecommittee module; the build is incomplete"
fi
say "adaptivecommittee module present"

cat <<EOF

Built: $OUT

Export it before running any scenario:

    export POC_CHAIND="$OUT"

Then, for a single scenario across three seeds:

    python3 "$HERE/scripts/epochrun_multiseed.py" \\
        "$HERE/configs/FP_A1_main_12h_c9_k6_b033_burst.yaml" \\
        --seeds 1,2,3 --artifacts-subdir pub_default_burst

Batches of twenty or more nodes also need a longer port-cleanup window:

    export ORPHAN_CLEANUP_TIMEOUT_S=180
EOF
