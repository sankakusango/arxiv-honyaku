#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   install_texlive.sh <year> <repository_url> [scheme]
# Example:
#   install_texlive.sh 2025 http://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final scheme-full

YEAR="$1"
REPOSITORY="$2"
SCHEME="${3:-scheme-full}"

INSTALL_ROOT="${TEXLIVE_INSTALL_ROOT:-/opt/texlive}"
INSTALL_DIR="${INSTALL_ROOT}/${YEAR}"
BIN_DIR="${INSTALL_DIR}/bin/x86_64-linux"

if [[ -x "${BIN_DIR}/pdflatex" ]]; then
  echo "[texlive:${YEAR}] already installed. skip."
  exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

curl -fsSL "${REPOSITORY}/install-tl-unx.tar.gz" -o "${WORK_DIR}/install-tl-unx.tar.gz"
tar -xzf "${WORK_DIR}/install-tl-unx.tar.gz" -C "${WORK_DIR}"

INSTALLER_DIR="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'install-tl-*' | head -n 1)"

cat > "${INSTALLER_DIR}/texlive.profile" <<EOF
selected_scheme ${SCHEME}
TEXDIR ${INSTALL_DIR}
option_doc 0
option_src 0
option_path 0
binary_x86_64-linux 1
EOF

"${INSTALLER_DIR}/install-tl" \
  --repository "${REPOSITORY}" \
  --profile "${INSTALLER_DIR}/texlive.profile"
