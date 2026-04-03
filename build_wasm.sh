#!/usr/bin/env bash
set -euo pipefail

# Build the WASM package and drop it into webapp/pkg/
echo "Building WASM (release)…"
wasm-pack build --target web --out-dir webapp/pkg --release

echo ""
echo "Done!  Serve the webapp/ folder, e.g.:"
echo "  cd webapp && python3 -m http.server 8080"
echo "Then open http://localhost:8080"
