#!/usr/bin/env bash
set -euo pipefail

# TeX Live install profile settings.
SELECTED_SCHEME="scheme-full"
BINARY_PLATFORM="x86_64-linux"

usage() {
  cat <<'EOF'
TeX Live (scheme-full) を ${TEXLIVE_INSTALL_ROOT:-/opt/texlive}/<year> にインストールします.

使い方:
  install_texlive.sh <year> <repository_url>
  install_texlive.sh -h | --help

引数:
  <year>            TeX Live のバージョン年 (例: 2025). インストール先のサブディレクトリ名になります.
  <repository_url>  install-tl-unx.tar.gz を配布している CTAN/tlnet のリポジトリ URL.

環境変数:
  TEXLIVE_INSTALL_ROOT  インストール先のルートディレクトリ (デフォルト: /opt/texlive).

例:
  install_texlive.sh 2025 http://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -eq 0 ]] && exit 1 || exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 1
fi

YEAR="$1"
REPOSITORY="$2"

INSTALL_ROOT="${TEXLIVE_INSTALL_ROOT:-/opt/texlive}"
INSTALL_DIR="${INSTALL_ROOT}/${YEAR}"
BIN_DIR="${INSTALL_DIR}/bin/${BINARY_PLATFORM}"

if [[ -x "${BIN_DIR}/pdflatex" ]]; then
  echo "[texlive:${YEAR}] already installed. skip."
  exit 0
fi

WORK_DIR="/tmp/texlive-installer-${YEAR}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

curl -fsSL "${REPOSITORY}/install-tl-unx.tar.gz" -o "${WORK_DIR}/install-tl-unx.tar.gz"
tar -xzf "${WORK_DIR}/install-tl-unx.tar.gz" -C "${WORK_DIR}"

INSTALLER_DIR="$(find "${WORK_DIR}" -maxdepth 1 -type d -name 'install-tl-*' | head -n 1)"

cat > "${INSTALLER_DIR}/texlive.profile" <<EOF
selected_scheme ${SELECTED_SCHEME}
TEXDIR ${INSTALL_DIR}
option_doc 0
option_src 0
option_path 0
binary_${BINARY_PLATFORM} 1
EOF

"${INSTALLER_DIR}/install-tl" \
  --repository "${REPOSITORY}" \
  --profile "${INSTALLER_DIR}/texlive.profile"

rm -rf "$WORK_DIR"
